const mongoose = require('mongoose');

const CharacterStateSchema = new mongoose.Schema({
   
    health: {
        type: Number,
        required: true
    },
    state: {
        type: String,
        required: true
    },
    attackAction: {
        type: String,
        required: false // Make attackAction nullable
    },
    attackDamage: {
        type: Number,
        required: true
    },
    heal: {
        type: Number,
        required: false, // Make heal nullable
        default: 0 // Default value if not provided
    },
    bleedingCount: {
        type: Number,
        required: false, // Make bleedingCount nullable
        default: 0 // Default value if not provided
    },
    bleedingDamage: {
        type: Number,
        required: false, // Make bleedingDamage nullable
        default: 0 // Default value if not provided
    },
    stunCount: {
        type: Number,
        required: false, // Make stunCount nullable
        default: 0 // Default value if not provided
    },

});

/* example:
{
    health: 100,
    state: 'normal',
    attackAction: 'fireball',
    attackDamage: 20
} */

module.exports = mongoose.model('CharacterState', CharacterStateSchema);