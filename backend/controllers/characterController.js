const Character = require('../models/CharacterModel');
const User = require('../models/UserModel');

// Get all characters for the logged-in user
exports.getAllCharacters = async (req, res) => {
    try {
        const user = await User.findById(req.user._id).populate('characters');
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }
        res.status(200).json(user.characters);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

// Get a character by ID for the logged-in user
exports.getCharacterById = async (req, res) => {
    try {
        const user = await User.findById(req.user._id).populate('characters');
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        const character = user.characters.find(c => c._id.toString() === req.params.id);
        if (!character) {
            return res.status(404).json({ message: 'Character not found' });
        }

        res.status(200).json(character);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

// Create a new character for the logged-in user
exports.createCharacter = async (req, res) => {
    const { characterName, avatar, class: characterClass, luck, attack, defense, vitality, attackType, attackDamage } = req.body;

    try {        // Validate the class field
        if (!['archer', 'mage', 'warrior'].includes(characterClass)) {
            return res.status(400).json({ message: 'Invalid class. Must be archer, mage, or warrior.' });
        }

        // Calculate maxHealth based on class and vitality
        const classHealthMap = {
            archer: 96,
            mage: 80,
            warrior: 120
        };
        const maxHealth = classHealthMap[characterClass] + vitality;

        const character = new Character({
            characterName,
            avatar,
            class: characterClass,
            luck,
            attack,
            defense,
            vitality,
            attackType,
            attackDamage,
            maxHealth
            
        });
        await character.save();

        const user = await User.findById(req.user._id);
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        user.characters.push(character._id);
        await user.save();

        res.status(201).json(character);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
};

// Delete a character by ID for the logged-in user
exports.deleteCharacter = async (req, res) => {
    try {
        const user = await User.findById(req.user._id);
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        const characterIndex = user.characters.findIndex(c => c.toString() === req.params.id);
        if (characterIndex === -1) {
            return res.status(404).json({ message: 'Character not found' });
        }

        user.characters.splice(characterIndex, 1);
        await user.save();

        await Character.findByIdAndDelete(req.params.id);

        res.status(200).json({ message: 'Character deleted successfully' });
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};