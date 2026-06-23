const express = require('express')
const router = express.Router()
const rateLimit = require('express-rate-limit')
const authController = require('../controllers/authController')
const { credentials, handleValidation } = require('../middlewares/validate')

// Throttle auth endpoints to slow down credential-stuffing / brute force.
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { message: 'Too many attempts, please try again later.' }
})

// Register endpoint
router.post('/register', authLimiter, credentials, handleValidation, authController.register)

// Login endpoint
router.post('/login', authLimiter, credentials, handleValidation, authController.login)

module.exports = router
