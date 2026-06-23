const jwt = require('jsonwebtoken')
const User = require('../models/UserModel')

const authenticate = async (req, res, next) => {
  const authHeader = req.header('Authorization')
  if (!authHeader) {
    return res.status(401).json({ message: 'Authorization header is missing' })
  }

  const token = authHeader.replace('Bearer ', '')
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET)
    const user = await User.findOne({ _id: decoded._id })

    if (!user) {
      console.log('User not found')
      throw new Error()
    }

    req.user = user; // Set the full user object in req.user
    next()
  } catch (error) {
    res.status(401).json({ message: 'Please authenticate.' })
  }
}

module.exports = authenticate
