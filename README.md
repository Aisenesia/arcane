# Arcane Gambit

A multiplayer game system combining a real-time game backend with a native
computer-vision library. This repository hosts two independently-buildable
modules that make up the project:

| Module | Path | Stack | Description |
|--------|------|-------|-------------|
| **Backend API** | [`backend/`](backend/) | Node.js · Express · MongoDB | REST API for accounts, characters, game sessions, and real-time character state. Serves the Unreal Engine, AR, and computer-vision clients. |
| **CV library** | [`cv/`](cv/) | C++ · OpenCV · ONNX Runtime | Native DLL that runs ONNX object-detection and classification models (dice/marker detection), exposed through a portable C ABI. |

Each module has its own README with setup and usage details:

- **Backend:** [`backend/README.MD`](backend/README.MD)
- **CV library:** [`cv/README.md`](cv/README.md)

## How the modules fit together

1. The **CV library** processes camera frames (e.g. from an ESP32-CAM or AR
   client) and returns detections/classifications through a thin C interface.
2. A client feeds those results to the **backend API**, which manages game
   sessions, characters, and turn-by-turn state.
3. Game clients (Unreal Engine, AR spectators) read and update session state
   through the backend's dedicated route groups.

## Repository layout

```
.
├── backend/   # Node.js/Express + MongoDB game API
├── cv/        # C++ OpenCV/ONNX detection & classification DLL
├── .gitattributes
└── .gitignore
```

## Getting started

See each module's README for prerequisites and build/run instructions:

```bash
# Backend
cd backend && npm install && npm run dev

# CV library (CMake)
cd cv && cmake -S . -B build && cmake --build build
```
