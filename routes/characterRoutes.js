const express = require('express');
const router = express.Router();
const characterController = require('../controllers/characterController');
const authenticate = require('../middlewares/authenticate');
const { character, handleValidation } = require('../middlewares/validate');

// Get all characters
router.get('/', authenticate, characterController.getAllCharacters);

// Get a character by ID
router.get('/:id', authenticate, characterController.getCharacterById);

// Create a new character
router.post('/', authenticate, character, handleValidation, characterController.createCharacter);

// Delete a character by ID
router.delete('/:id', authenticate, characterController.deleteCharacter);

module.exports = router;