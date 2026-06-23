const express = require('express')
const mongoose = require('mongoose')
const cors = require('cors')
const helmet = require('helmet')
const rateLimit = require('express-rate-limit')
const dotenv = require('dotenv')
const os = require('os')
const fs = require('fs')
const https = require('https')

const User = require('./models/UserModel')
const helpers = require('./utils/helpers')
const authenticate = require('./middlewares/authenticate')
const errorHandler = require('./middlewares/errorHandler')

// Load environment variables from .env file
dotenv.config()

const app = express()

// Security & parsing middleware
app.use(helmet())
app.use(cors())
app.use(express.json())

// Basic global rate limit to protect against abuse.
app.use(
  rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 1000,
    standardHeaders: true,
    legacyHeaders: false
  })
)

// Routes
const userRoutes = require('./routes/userRoutes')
const authRoutes = require('./routes/authRoutes')
const characterRoutes = require('./routes/characterRoutes')
const arRoutes = require('./routes/arRoutes')
const unrealRoutes = require('./routes/unrealRoutes')
const cvRoutes = require('./routes/cvRoutes')

app.use('/api/users', userRoutes)
app.use('/api/auth', authRoutes)
app.use('/api/characters', authenticate, characterRoutes)
app.use('/api/ar', arRoutes)
app.use('/api/unreal', unrealRoutes)
app.use('/api/cv', authenticate, cvRoutes)

// Error handling middleware (must be registered last)
app.use(errorHandler)

const PORT = process.env.PORT || 3001
const useRemote = process.env.USE_REMOTE_DB === 'true'
const MONGODB_URI = useRemote
  ? process.env.MONGODB_URI
  : process.env.LOCAL_MONGODB_URI

mongoose.connect(MONGODB_URI, {})

const connection = mongoose.connection
connection.once('open', async () => {
  // Avoid printing the full URI since it may contain credentials.
  console.log(`MongoDB connected (${useRemote ? 'remote' : 'local'})`)
  await ensureAdminAccount()
})

// Ensure a single admin account exists. The password comes from ADMIN_PASSWORD,
// or a random one that is printed once on first creation.
const ensureAdminAccount = async () => {
  try {
    const adminEmail = process.env.ADMIN_EMAIL || 'admin'

    const existing = await User.findOne({ email: adminEmail })
    if (existing) return

    const password = process.env.ADMIN_PASSWORD || helpers.generateRandomString(16)
    await new User({
      email: adminEmail,
      isAdmin: true,
      isActive: true,
      password
    }).save()

    console.log('Admin account created.')
    if (!process.env.ADMIN_PASSWORD) {
      console.log(`Generated admin password: ${password}`)
    }
  } catch (error) {
    console.error('Error creating admin account:', error)
  }
}

// Collect non-internal IPv4 addresses so the operator can reach the server
// from other devices on the local network.
const getNetworkInterfaces = () => {
  const interfaces = os.networkInterfaces()
  const addresses = []

  for (const interfaceName in interfaces) {
    for (const iface of interfaces[interfaceName]) {
      if (!iface.internal && iface.family === 'IPv4') {
        addresses.push({ interface: interfaceName, address: iface.address })
      }
    }
  }

  return addresses
}

const displayServerInfo = port => {
  const networkInterfaces = getNetworkInterfaces()

  console.log('\n=== SERVER STARTED ===')
  console.log(`Port: ${port}`)
  console.log('\nLocal access:')
  console.log(`  http://localhost:${port}`)

  if (networkInterfaces.length > 0) {
    console.log('\nNetwork access:')
    networkInterfaces.forEach(({ interface: interfaceName, address }) => {
      console.log(`  http://${address}:${port} (${interfaceName})`)
    })
  }
  console.log('======================\n')
}

// Set TLS_KEY_PATH and TLS_CERT_PATH to serve over HTTPS; otherwise HTTP is used.
const keyPath = process.env.TLS_KEY_PATH
const certPath = process.env.TLS_CERT_PATH

if (keyPath && certPath) {
  const options = {
    key: fs.readFileSync(keyPath),
    cert: fs.readFileSync(certPath)
  }
  https.createServer(options, app).listen(PORT, '0.0.0.0', () => {
    console.log(`HTTPS server running on port ${PORT}`)
    displayServerInfo(PORT)
  })
} else {
  app.listen(PORT, '0.0.0.0', () => {
    displayServerInfo(PORT)
  })
}
