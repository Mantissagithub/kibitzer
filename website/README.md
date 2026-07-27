# Kibitzer website

The website is a separate Vite application inside the Kibitzer repository. Its
logbook, tournament games, ratings, and figures are generated from the project
files, so those records stay in one place.

```bash
npm install
npm run dev
```

Run the complete local checks with:

```bash
npm test
npm run lint
npm run typecheck
npm run check:copy
npm run build
```

`npm run content` rebuilds the generated website data from `../LOGBOOK.md`, the
official Elo PGN, and the experiment reports. Generated output is intentionally
ignored by Git and is rebuilt before development, type checking, and production
builds.
