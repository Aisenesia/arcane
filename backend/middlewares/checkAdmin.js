const checkAdmin = (req, res, next) => {
  // `authenticate` runs first and attaches the full user document to req.user.
  if (!req.user || !req.user.isAdmin) {
    return res.status(403).json({ message: 'Access denied. Admins only.' })
  }
  next()
}

module.exports = checkAdmin
