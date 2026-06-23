const mongoose = require('mongoose')
const CharacterState = require('./CharacterState') // Import CharacterState model

const GameSchema = new mongoose.Schema({
  sessionId: {
    type: Number
  },
  timeStamp: {
    type: Date,
    default: Date.now // Automatically set the timestamp when the game is created
  },
  gameStatus: {
    type: String,
    required: true,
    enum: ['ongoing', 'started', 'finished'] // Valid game states
  },
  currentTurnCharacterId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    ref: 'Character' // Reference to the User model
  },
  roomCode: {
    type: String,
    required: false,
    unique: true,
    length: 6
  },
  users: [
    {
      userid: {
        type: mongoose.Schema.Types.ObjectId,
        required: true,
        ref: 'User' // Reference to the User model
      },
      characterState: {
        type: mongoose.Schema.Types.ObjectId,
        required: false,
        ref: 'CharacterState' // Reference to the CharacterState model
      },
      characterName: {
        type: String,
        required: true
      },
      class: {
        type: String,
        required: true,
        enum: ['archer', 'mage', 'warrior']
      },
      avatar: {
        type: String,
        required: true
      }
    }
  ],
  spectators: [
    {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User' // Reference to the User model
    }
  ]
})

module.exports = mongoose.model('Game', GameSchema)
