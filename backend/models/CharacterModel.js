const mongoose = require('mongoose')

const CharacterSchema = new mongoose.Schema({
  characterName: {
    type: String,
    required: true
  },
  avatar: {
    type: String,
    required: true
  },
  class: {
    type: String,
    required: true,
    enum: ['archer', 'mage', 'warrior'] // Restrict values to these options
  },
  luck: {
    type: Number,
    required: true
  },
  attack: {
    type: Number,
    required: true
  },
  defense: {
    type: Number,
    required: true
  },
  vitality: {
    type: Number,
    required: true
  },
  maxHealth: {
    type: Number,
    required: false
  },
  characterState: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'CharacterState' // Reference to the CharacterState model
  }
})

module.exports = mongoose.model('Character', CharacterSchema)

/*

maybe we should change this to be a characterAction model and following as the character model?
TODO: ask group

luck: {
    type: Number,
    required: true
},
attack: {
    type: Number,
    required: true
},
defense: {
    type: Number,
    required: true
},
attackType: {
    type: String,
    required: true
},
attackDamage: {
    type: Number,
    required: true
},
*/
