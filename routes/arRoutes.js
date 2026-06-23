const express = require('express');
const router = express.Router();
const gameController = require('../controllers/gameController');
const authenticate = require('../middlewares/authenticate');

// Add a spectator to a session by roomCode
router.post('/room/:roomCode/add-spectator', authenticate, gameController.addSpectator);

// Remove a spectator from a session
router.delete('/:sessionId/remove-spectator', authenticate, gameController.removeSpectator);

// Get session details
router.get('/:sessionId', authenticate, gameController.getSession);

module.exports = router;

