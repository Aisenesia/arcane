const jwt = require('jsonwebtoken')
const bcrypt = require('bcryptjs')
const User = require('../models/UserModel')

// Token lifetime is configurable; defaults to 7 days.
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d'

const signToken = user =>
  jwt.sign({ _id: user._id }, process.env.JWT_SECRET, {
    expiresIn: JWT_EXPIRES_IN
  })

exports.register = async (req, res) => {
  const { email, password } = req.body

  try {
    const existingUser = await User.findOne({ email })
    if (existingUser) {
      return res.status(400).json({ message: 'Email already in use' })
    }

    const user = new User({ email, password })
    await user.save()

    res.status(201).json({ token: signToken(user) })
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

exports.login = async (req, res) => {
  const { email, password } = req.body

  try {
    const user = await User.findOne({ email })
    if (!user) {
      // Same response as a bad password to avoid leaking which emails exist.
      return res.status(401).json({ message: 'Invalid email or password' })
    }

    if (!user.isActive) {
      return res.status(403).json({
        message: 'Account is not activated. Please verify your email.'
      })
    }

    const isMatch = await bcrypt.compare(password, user.password)
    if (!isMatch) {
      return res.status(401).json({ message: 'Invalid email or password' })
    }

    res.status(200).json({ token: signToken(user) })
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}
