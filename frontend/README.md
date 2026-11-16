# ModelMind Frontend

A React + Vite starter interface for the ModelMind Python ML agent.

## Run locally

```bash
npm install
npm run dev
```

Open the local URL printed by Vite (usually http://localhost:5173).

## Build check

```bash
npm run build
```

## Current scope

- Responsive workspace shell and sidebar navigation
- Light mode by default, dark mode toggle persisted in localStorage
- Overview explaining the ModelMind workflow
- New Analysis page with CSV file selection and optional target/goal fields
- Informational Experiments, Optimization and About pages

The interface is intentionally not wired to the Python engine yet. API integration is the next stage; do not present the file-selection action as a completed model run.
