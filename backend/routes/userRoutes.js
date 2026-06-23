const express = require('express')
const router = express.Router()
const userController = require('../controllers/userController')
const authenticate = require('../middlewares/authenticate')
const checkAdmin = require('../middlewares/checkAdmin')
const {
  emailOnly,
  passwordReset,
  handleValidation
} = require('../middlewares/validate')

// Get all users (Admin only)
router.get('/', authenticate, checkAdmin, userController.getAllUsers)

// Get user by ID (Authenticated users)
router.get('/:id', authenticate, userController.getUserById)

// Create a new user (Admin only)
router.post('/', authenticate, checkAdmin, userController.createUser)

// Update a user (Admin only)
router.put('/:id', authenticate, checkAdmin, userController.updateUser)

// Delete a user (Admin only)
router.delete('/:id', authenticate, checkAdmin, userController.deleteUser)

// Password recovery
router.post('/forgot-password', emailOnly, handleValidation, userController.forgotPassword)
router.post('/reset-password/:token', passwordReset, handleValidation, userController.resetPassword)
router.get('/verify-code/:token', userController.verifyCode)

module.exports = router
