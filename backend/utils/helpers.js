// Utility functions for the server

/**
 * Generates a random string of specified length
 * @param {number} length - Length of the random string
 * @returns {string} - Random string
 */
function generateRandomString (length) {
  const characters =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
  let result = ''
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length))
  }
  return result
}

/**
 * Checks if an object is empty
 * @param {object} obj - Object to check
 * @returns {boolean} - True if object is empty, false otherwise
 */
function isEmptyObject (obj) {
  return Object.keys(obj).length === 0 && obj.constructor === Object
}

module.exports = {
  generateRandomString,
  isEmptyObject
}
