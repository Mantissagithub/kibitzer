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

To use the live play page, copy `.env.example` to `.env.local` and run the
inference container from the repository root. `VITE_KIBITZER_API_URL` must
point at that service. The browser keeps the game record and sends only the UCI
move history plus the selected search budget.

`npm run content` rebuilds the generated website data from `../LOGBOOK.md`, the
official Elo PGN, and the experiment reports. Generated output is intentionally
ignored by Git and is rebuilt before development, type checking, and production
builds.

Vercel production deploys must use the local prebuilt flow because those source
files live above `website/` and are not present in a remote build upload:

```bash
vercel pull --yes --environment=production
vercel build --prod
vercel deploy --prebuilt --prod --yes
```
