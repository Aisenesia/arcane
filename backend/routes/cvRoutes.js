const express = require('express');
const router = express.Router();
const gameController = require('../controllers/gameController');
const authenticate = require('../middlewares/authenticate');

router.post('/create', authenticate, gameController.cvCreateSession);

router.post('/:sessionId/join', authenticate, gameController.joinSession);

module.exports = router;
