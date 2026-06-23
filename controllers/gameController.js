const Game = require('../models/GameModel')
const User = require('../models/UserModel') // Ensure User model is imported
const Character = require('../models/CharacterModel')
const CharacterState = require('../models/CharacterState')

// Create a new game session
exports.createSession = async (req, res) => {
  try {
    const { gameStatus, characterState, characterId } = req.body
    const userId = req.user.id // Extract userId from token

    // create character state as the character is created
    const characterStateData = new CharacterState(characterState)
    await characterStateData.save() // Save the character state to the database

    const currentTurnCharacterStateId = characterStateData._id // Use the saved character state ID

    // Fetch character details
    const character = await Character.findById(characterId)
    if (!character)
      return res.status(404).json({ message: 'Character not found' })

    const newGame = new Game({
      gameStatus,
      currentTurnCharacterId: characterId, // Update to use character state
      users: [
        {
          userid: userId,
          characterState: currentTurnCharacterStateId,
          characterName: character.characterName,
          class: character.class,
          avatar: character.avatar
        }
      ], // Automatically add the creator to the game
      spectators: []
    })
    await newGame.save()
    res.status(201).json(newGame)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

exports.cvCreateSession = async (req, res) => {
  try {
    const { characterId } = req.body
    const userId = req.user.id // Extract userId from token

    const character = await Character.findById(characterId)
    if (!character)
      return res.status(404).json({ message: 'Character not found' })

    // Create character state based on character properties
    const characterStateData = new CharacterState({
      health: character.maxHealth,
      state: 'idle',
      attackAction: '',
      attackDamage: 0
    })
    await characterStateData.save()

    // Generate a unique 6-digit room code
    let roomCode
    let isUnique = false
    while (!isUnique) {
      roomCode = Math.floor(100000 + Math.random() * 900000).toString()
      const existing = await Game.findOne({ roomCode })
      if (!existing) isUnique = true
    }

    const newGame = new Game({
      gameStatus: 'started',
      currentTurnCharacterId: characterId, // Update to use character state
      roomCode,
      users: [
        {
          characterState: characterStateData._id, // Use the created character state
          userid: userId,
          characterName: character.characterName,
          class: character.class,
          avatar: character.avatar
        }
      ], // Automatically add the creator to the game
      spectators: []
    })
    await newGame.save()
    res.status(201).json(newGame)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

exports.joinSession = async (req, res) => {
  try {
    const { sessionId } = req.params
    const { characterId } = req.body
    const userId = req.user.id // Extract userId from token

    const game = await Game.findById(sessionId).populate('users.characterState')
    if (!game) return res.status(404).json({ message: 'Session not found' })

    // Fetch character details
    const character = await Character.findById(characterId)
    if (!character)
      return res.status(404).json({ message: 'Character not found' })

    // Check if character is already in the session
    const existingUser = game.users.find(
      user =>
        user.characterName === character.characterName &&
        user.class === character.class &&
        user.avatar === character.avatar
    )
    if (existingUser) {
      return res
        .status(400)
        .json({ message: 'This character is already in the session' })
    }

    // Create character state based on character properties
    const characterStateData = new CharacterState({
      health: character.maxHealth,
      state: 'idle',
      attackAction: '',
      attackDamage: 0
    })
    await characterStateData.save()

    game.users.push({
      userid: userId,
      characterState: characterStateData._id,
      characterName: character.characterName,
      class: character.class,
      avatar: character.avatar
    })
    await game.save()
    res.status(200).json(game)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

exports.getSession = async (req, res) => {
  try {
    const { sessionId } = req.params
    const game = await Game.findById(sessionId).populate('users.characterState')
    if (!game) return res.status(404).json({ message: 'Session not found' })

    // Add maxHealth to each user in the response
    const usersWithMaxHealth = await Promise.all(
      game.users.map(async user => {
        // Try to get maxHealth from characterState, fallback to Character if needed
        let maxHealth = null
        if (user.characterState && user.characterState.health) {
          maxHealth = user.characterState.health
        } else {
          // Fallback: fetch Character by name/class/avatar
          const character = await Character.findOne({
            characterName: user.characterName,
            class: user.class,
            avatar: user.avatar
          })
          if (character) maxHealth = character.maxHealth
        }
        return { ...user.toObject(), maxHealth }
      })
    )

    const gameObj = game.toObject()
    gameObj.users = usersWithMaxHealth
    res.status(200).json(gameObj)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}


exports.getSessionUnreal = async (req, res) => {
  try {
    const { sessionId } = req.params
    const game = await Game.findById(sessionId)
      .populate('users.characterState')
    if (!game) return res.status(404).json({ message: 'Session not found' })

    // Add character details (vitality, attack, defense, etc.) to each user in the response
    const usersWithCharacterDetails = await Promise.all(
      game.users.map(async user => {
        // Find the character that matches this user's character in the game
        const character = await Character.findOne({
          characterName: user.characterName,
          class: user.class,
          avatar: user.avatar
        })
        
        const userObj = user.toObject()
        if (character) {
          userObj.character = {
            _id: character._id,
            characterName: character.characterName,
            avatar: character.avatar,
            class: character.class,
            luck: character.luck,
            attack: character.attack,
            defense: character.defense,
            vitality: character.vitality,
            maxHealth: character.maxHealth
          }
        }
        
        return userObj
      })
    )

    const gameObj = game.toObject()
    gameObj.users = usersWithCharacterDetails
    res.status(200).json(gameObj)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

// End a game session
exports.endSession = async (req, res) => {
  try {
    const { sessionId } = req.params
    const game = await Game.findById(sessionId)
    if (!game) return res.status(404).json({ message: 'Session not found' })

    // Delete all character states associated with the session
    const characterStateIds = game.users.map(user => user.characterState)
    await CharacterState.deleteMany({ _id: { $in: characterStateIds } })

    game.gameStatus = 'finished'
    await game.save()
    res.status(200).json(game)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

exports.deleteSession = async (req, res) => {
  try {
    const { sessionId } = req.params
    // find the session by id, if token acquirer is one of the players or the token acquirer is the admin, delete the session
    const game = await Game.findById(sessionId)
    if (!game) return res.status(404).json({ message: 'Session not found' })

    // Delete all character states associated with the session
    const characterStateIds = game.users.map(user => user.characterState)
    await CharacterState.deleteMany({ _id: { $in: characterStateIds } })

    // Check if the user is a player or admin
    const userId = req.user.id // Extract userId from token
    const isPlayer = game.users.some(user => user.userid.toString() === userId)
    const isAdmin = req.user.isAdmin // Assuming you have a role field in your user model

    if (isPlayer || isAdmin) {
      await Game.deleteOne({ _id: sessionId })
      res.status(200).json({ message: 'Session deleted successfully' })
    } else {
      res
        .status(403)
        .json({ message: 'You do not have permission to delete this session' })
    }
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

// Add a player to a session
exports.addPlayer = async (req, res) => {
  try {
    const { sessionId } = req.params
    const { characterState, characterId } = req.body
    const userId = req.user.id // Extract userId from token

    const characterStateData = new CharacterState(characterState)
    await characterStateData.save() // Save the character state to the database

    const game = await Game.findById(sessionId).populate('users.characterState')
    if (!game) return res.status(404).json({ message: 'Session not found' })

    // Fetch character details
    const character = await Character.findById(characterId)
    if (!character)
      return res.status(404).json({ message: 'Character not found' })

    // Check if character is already in the session
    const existingUser = game.users.find(
      user =>
        user.characterName === character.characterName &&
        user.class === character.class &&
        user.avatar === character.avatar
    )
    if (existingUser) {
      // Delete the character state we just created since we're not using it
      await CharacterState.deleteOne({ _id: characterStateData._id })
      return res
        .status(400)
        .json({ message: 'This character is already in the session' })
    }

    game.users.push({
      userid: userId,
      characterState: characterStateData._id,
      characterName: character.characterName,
      class: character.class,
      avatar: character.avatar
    })
    await game.save()
    res.status(200).json(game)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

exports.removePlayer = async (req, res) => {
  try {
    const { sessionId } = req.params
    const userId = req.user.id // Extract userId from token

    const game = await Game.findById(sessionId)
    if (!game) return res.status(404).json({ message: 'Session not found' })

    game.users = game.users.filter(user => user.userid.toString() !== userId)
    await game.save()
    res.status(200).json(game)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

// Add a spectator to a session by roomCode
exports.addSpectator = async (req, res) => {
  try {
    const { roomCode } = req.params
    const userId = req.user.id // Extract userId from token

    // Find the game by roomCode instead of sessionId
    const game = await Game.findOne({ roomCode })
    if (!game) return res.status(404).json({ message: 'Session not found' })

    // Ensure the user is not already a spectator
    if (game.spectators.includes(userId)) {
      return res.status(400).json({ message: 'User is already a spectator' })
    }

    game.spectators.push(userId)
    await game.save()
    res.status(200).json(game)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

// Remove a spectator from a session
exports.removeSpectator = async (req, res) => {
  try {
    const { sessionId } = req.params
    const userId = req.user.id // Extract userId from token

    const game = await Game.findById(sessionId)
    if (!game) return res.status(404).json({ message: 'Session not found' })

    game.spectators = game.spectators.filter(s => s.toString() !== userId)
    await game.save()
    res.status(200).json(game)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

// Update a player's character state
exports.updateCharacterState = async (req, res) => {
  try {
    const { sessionId } = req.params
    const { characterState } = req.body
    const userId = req.user.id // Extract userId from token

    const game = await Game.findById(sessionId).populate('users.characterState')
    if (!game) return res.status(404).json({ message: 'Session not found' })

    // Find the user in the game
    const user = game.users.find(user => user.userid.toString() === userId)
    if (!user)
      return res.status(404).json({ message: 'User not found in this session' })

    // Update the character state
    user.characterState = characterState
    await game.save()
    res.status(200).json(game)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}

exports.updateCharacterStates = async (req, res) => {
  try {
    const { sessionId } = req.params
    const { currentTurnCharacterId, characterStates } = req.body // Expecting currentTurnCharacterId and array/object of character states
    const userId = req.user.id // Extract userId from token

    const game = await Game.findById(sessionId).populate('users.characterState')
    if (!game) return res.status(404).json({ message: 'Session not found' })

    // Check if the user is part of the game
    const user = game.users.find(user => user.userid.toString() === userId)
    if (!user)
      return res.status(404).json({ message: 'User not found in this session' })

    // Update currentTurnCharacterId if provided
    if (currentTurnCharacterId) {
      game.currentTurnCharacterId = currentTurnCharacterId
    }

    // Handle character states - can be an array or object with character state properties
    if (characterStates) {
      // If characterStates is an array, iterate through each state
      if (Array.isArray(characterStates)) {
        for (const state of characterStates) {
          const characterState = await CharacterState.findById(state._id)
          if (characterState) {
            Object.assign(characterState, state) // Update the character state with new values
            await characterState.save()
          }
        }
      } else if (typeof characterStates === 'object') {
        // If characterStates is an object with multiple character states
        for (const characterStateId in characterStates) {
          const stateData = characterStates[characterStateId]
          const characterState = await CharacterState.findById(characterStateId)
          if (characterState) {
            Object.assign(characterState, stateData) // Update the character state with new values
            await characterState.save()
          }
        }
      }
    }

    await game.save() // Save the game to update currentTurnCharacterId
    res.status(200).json(game)
  } catch (error) {
    res.status(500).json({ message: error.message })
  }
}
