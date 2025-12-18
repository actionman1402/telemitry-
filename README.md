# Telemitry Viewer

A desktop viewer for Motion IQ telemetry that you can package as a macOS DMG. The app ships with mock data and components for GPS visualization, IMU plots, battery and speed charts, and tabular event review—similar to a BYB-style interface.

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

3) **Run the app in development with hot reload**

```bash
npm run dev
```

- Vite serves the renderer at `http://localhost:5173`.
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

- `src/` – React renderer with GPS trace, IMU and speed charts, and raw sample table.
- `electron/` – Electron main and preload scripts used for packaging.
- `data/mockTelemetry.json` – Example Motion IQ telemetry feed used at runtime.
- `vite.config.ts` – Vite configuration for the renderer build.

## Wiring to real data

Replace `data/mockTelemetry.json` with a loader that reads Motion IQ session exports or streams live samples from your gateway. The UI filters by event type and aggregates averages, so you can drop in your parser while keeping the same component contracts.

## Importing telemetry files

- Use **Import data** in the toolbar to load a Motion IQ JSON export at runtime.
- Files should contain an array of samples shaped like `data/mockTelemetry.json` (timestamp, gps, imu, battery, event).
- Imports reset the filter to **All events** and fall back to the bundled mock dataset if parsing fails.
