# Arcane Gambit — Game Backend API

A Node.js/Express + MongoDB backend for a multiplayer game system with user
accounts, character management, game sessions, and real-time character-state
updates. It exposes dedicated route groups for the game's different clients:
**Unreal Engine**, an **AR** spectator app, and a **computer-vision (CV)** client.

## Tech Stack

- **Runtime:** Node.js (18+), Express 5
- **Database:** MongoDB via Mongoose
- **Auth:** JWT (`jsonwebtoken`) with `bcryptjs` password hashing
- **Security:** `helmet`, `express-rate-limit`, `express-validator`
- **Email:** `nodemailer` (password recovery)

## Project Structure

```
.
├── controllers/   # Request handlers (auth, users, characters, games)
├── middlewares/   # authenticate, checkAdmin, validation, email, error handling
├── models/        # Mongoose schemas (User, Character, Game, CharacterState, RecoveryToken)
├── routes/        # Express routers, one per resource
├── utils/         # Helper functions
├── server.js      # App entry point: middleware, route mounting, DB + server bootstrap
└── .env.example   # Documented environment variables
```

## Getting Started

### Prerequisites

- Node.js 18 or higher
- A MongoDB instance (local or remote)

### Installation

```bash
git clone https://github.com/AhmetBahaCepni/ArcaneGambit
cd backend
npm install
cp .env.example .env   # then edit .env
npm run dev            # or: npm start
```

On first boot an `admin` account is created. Set `ADMIN_PASSWORD` in `.env`, or
leave it blank to have a random password generated and printed to the console once.

## Configuration

All configuration is via environment variables (see [.env.example](.env.example)):

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `PORT` | Server port | No | `3001` |
| `USE_REMOTE_DB` | Use `MONGODB_URI` instead of `LOCAL_MONGODB_URI` | No | `false` |
| `LOCAL_MONGODB_URI` | Local MongoDB connection string | Yes* | – |
| `MONGODB_URI` | Remote MongoDB connection string | Yes* | – |
| `JWT_SECRET` | Secret used to sign JWTs | **Yes** | – |
| `JWT_EXPIRES_IN` | Token lifetime | No | `7d` |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | Bootstrap admin credentials | No | `admin` / random |
| `SMTP_*` | SMTP credentials for password-recovery email | No | – |
| `TLS_KEY_PATH` / `TLS_CERT_PATH` | Enable HTTPS when both are set | No | – |

\* At least one MongoDB URI is required, matching `USE_REMOTE_DB`.

## Authentication

Most endpoints require a JWT in the `Authorization` header:

```
Authorization: Bearer <token>
```

Obtain a token from `POST /api/auth/register` or `POST /api/auth/login`.
Admin-only endpoints additionally require the account's `isAdmin` flag.

## API Reference

Base URL: `http://<host>:<port>/api`

### Auth — `/api/auth`
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/register` | – | Register and receive a JWT |
| POST | `/login` | – | Log in and receive a JWT |

### Users — `/api/users`
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | Admin | List all users |
| GET | `/:id` | User | Get a user by ID |
| POST | `/` | Admin | Create a user |
| PUT | `/:id` | Admin | Update a user |
| DELETE | `/:id` | Admin | Delete a user |
| POST | `/forgot-password` | – | Email a recovery code |
| POST | `/reset-password/:token` | – | Reset password with code |
| GET | `/verify-code/:token?email=` | – | Validate a recovery code |

### Characters — `/api/characters` (all require auth)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | List the current user's characters |
| GET | `/:id` | Get one of the user's characters |
| POST | `/` | Create a character |
| DELETE | `/:id` | Delete a character |

### Game sessions — CV client `/api/cv` (all require auth)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/create` | Create a session (generates a 6-digit room code) |
| POST | `/:sessionId/join` | Join a session with a character |

### Game sessions — Unreal Engine `/api/unreal` (all require auth)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/create` | Create a session |
| GET | `/:sessionId` | Get session details (with full character stats) |
| POST | `/:sessionId/add-player` | Add a player |
| DELETE | `/:sessionId/remove-player` | Remove a player |
| PUT | `/:sessionId/update-states` | Update character states + current turn |
| PUT | `/end/:sessionId` | Mark a session finished |
| DELETE | `/:sessionId` | Delete a session (player or admin) |

### AR spectators — `/api/ar` (all require auth)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/:sessionId` | Get session details |
| POST | `/room/:roomCode/add-spectator` | Add a spectator by room code |
| DELETE | `/:sessionId/remove-spectator` | Remove a spectator |

## Domain Model

**Character classes** determine base health (`maxHealth = base + vitality`):

| Class | Base health |
|-------|-------------|
| Warrior | 120 |
| Archer | 96 |
| Mage | 80 |

**Character stats:** `attack`, `defense`, `vitality`, `luck`.

### Game session flow
1. A host creates a session (`/api/cv/create` or `/api/unreal/create`); a room code is generated.
2. Players join with a character; a `CharacterState` is created per player.
3. Clients sync state via `PUT /api/unreal/:sessionId/update-states`.
4. AR clients join as spectators via room code.
5. The session is ended or deleted on completion.

## Security Notes

- Passwords are hashed with bcrypt (`pre('save')` hook); plaintext is never stored.
- JWT-protected routes; admin routes gated by an `isAdmin` check.
- `helmet` sets secure HTTP headers; `express-rate-limit` throttles requests
  (stricter limits on the auth endpoints).
- Request bodies are validated with `express-validator`, which also closes the
  NoSQL-injection vector on email/password fields.
- Secrets live only in `.env` (git-ignored); `.env.example` documents the shape.

## Network Access

The server binds to `0.0.0.0`, so it is reachable from other devices on the LAN
(the console prints the available addresses on startup). On Windows you may need
an inbound firewall rule for the chosen port.
