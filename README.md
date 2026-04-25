# Telemitry Viewer

A desktop viewer for Motion IQ telemetry that you can package as a macOS DMG. The renderer is now implemented in plain HTML/CSS/JavaScript (no React runtime required) and includes BYB-style cards, charts, GPS rows, event filtering, and file import support.

## Quick start

1) **Install prerequisites**

- Node.js 18+ and npm installed on your machine
- Internet access to reach the npm registry (private networks may need a proxy)

2) **Install dependencies**

```bash
npm install
```

If you see a 403 when fetching packages, try:

- `npm config set registry https://registry.npmjs.org/` (reset to the public registry)
- Check any corporate proxy settings or VPN requirements

3) **Run the app in development**

```bash
npm run dev
```

- Vite serves the HTML renderer at `http://localhost:5173`.
- Electron waits for the dev server via `wait-on` and opens the desktop window automatically.

4) **Build production assets (renderer + Electron)**

```bash
npm run build
```

5) **Export a macOS DMG** (requires macOS; signing/notarization optional):

```bash
npm run dist
```

Electron Builder writes the DMG to `release/Telemitry Viewer-<version>-mac.dmg` so you can ship or test it in Finder. Use `npm run package` to create an unpacked directory for inspection instead of a DMG.

## Project structure

- `index.html` – Complete telemetry viewer UI (toolbar, metrics, charts, GPS panel, data table, import flow).
- `electron/` – Electron main and preload scripts used for packaging.
- `data/mockTelemetry.json` – Example Motion IQ telemetry feed shape for your own exports.
- `vite.config.ts` – Vite configuration used for renderer build + Electron integration.

## Importing telemetry files

- Use **Import telemetry JSON** in the toolbar to load a Motion IQ JSON export at runtime.
- Files should contain a non-empty sample array with fields: `timestamp`, `gps.lat`, `gps.lon`, `imu`, `battery`, `speed`, `event`.
- Failed imports show an inline error and automatically fall back to bundled mock data.
