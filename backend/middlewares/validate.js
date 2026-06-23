const { body, validationResult } = require('express-validator')

// Collects validation errors from the chains below and returns a 400 if any failed.
const handleValidation = (req, res, next) => {
  const errors = validationResult(req)
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() })
  }
  next()
}

// Requiring `.isString()`/`.isEmail()` also closes the NoSQL-injection hole where a
// client sends e.g. { "email": { "$gt": "" } } to bypass the lookup.
const credentials = [
  body('email').isEmail().withMessage('A valid email is required'),
  body('password')
    .isString()
    .isLength({ min: 8 })
    .withMessage('Password must be at least 8 characters')
]

const emailOnly = [body('email').isEmail().withMessage('A valid email is required')]

const passwordReset = [
  body('newPassword')
    .isString()
    .isLength({ min: 8 })
    .withMessage('Password must be at least 8 characters')
]

const character = [
  body('characterName').isString().trim().notEmpty(),
  body('class').isIn(['archer', 'mage', 'warrior']),
  body('luck').optional().isInt({ min: 0 }),
  body('attack').optional().isInt({ min: 0 }),
  body('defense').optional().isInt({ min: 0 }),
  body('vitality').optional().isInt({ min: 0 })
]

module.exports = {
  handleValidation,
  credentials,
  emailOnly,
  passwordReset,
  character
}
