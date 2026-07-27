#!/usr/bin/env node
// Generates src/generated/{logbook.mdx,logbook-index.json,leaderboard.json} and
// public/generated/{logbook-assets/**,official-elo-clean.pgn} from
// repo-root source artifacts. Must run with cwd = website/.

import { promises as fs } from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const WEBSITE_ROOT = process.cwd();
const REPO_ROOT = path.resolve(WEBSITE_ROOT, "..");

const LOGBOOK_PATH = path.join(REPO_ROOT, "LOGBOOK.md");
const PGN_PATH = path.join(REPO_ROOT, "reports/official_elo/official_elo_s512_gpp40_clean.pgn");
const RATINGS_PATH = path.join(REPO_ROOT, "reports/official_elo/ratings_clean.txt");

const SRC_GENERATED = path.join(WEBSITE_ROOT, "src/generated");
const PUBLIC_GENERATED = path.join(WEBSITE_ROOT, "public/generated");
const LOGBOOK_ASSETS_DIR = path.join(PUBLIC_GENERATED, "logbook-assets");

async function readText(p) {
  return fs.readFile(p, "utf8");
}

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function readPngDimensions(buf, label = "PNG asset") {
  if (
    buf.length < 24 ||
    buf.readUInt32BE(0) !== 0x89504e47 ||
    buf.readUInt32BE(4) !== 0x0d0a1a0a
  ) {
    throw new Error(`${label} is not a valid PNG file`);
  }
  const width = buf.readUInt32BE(16);
  const height = buf.readUInt32BE(20);
  return { width, height };
}

function slugify(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function parseDecisions(markdown) {
  const lines = markdown.split("\n");
  const decisions = [];
  let currentPhase = null;
  for (const line of lines) {
    const phaseMatch = line.match(/^## Phase (.+)$/);
    if (phaseMatch) {
      currentPhase = phaseMatch[1].trim();
      continue;
    }
    const decisionMatch = line.match(/^### D(\d+)\.\s+(.+)$/);
    if (decisionMatch) {
      const number = Number(decisionMatch[1]);
      const title = decisionMatch[2].trim();
      decisions.push({
        id: `D${number}`,
        number,
        title,
        phase: currentPhase,
        slug: slugify(`D${number} ${title}`),
      });
    }
  }
  return decisions;
}

// Reject unsafe paths (absolute, protocol-based, or escaping the repo root).
function classifyImagePath(rawPath) {
  if (/^[a-z][a-z0-9+.-]*:/i.test(rawPath) || rawPath.startsWith("//")) {
    return { safe: false, reason: "protocol-based path" };
  }
  if (path.isAbsolute(rawPath)) {
    return { safe: false, reason: "absolute path" };
  }
  const resolved = path.resolve(REPO_ROOT, rawPath);
  const relativeToRoot = path.relative(REPO_ROOT, resolved);
  if (relativeToRoot.startsWith("..") || path.isAbsolute(relativeToRoot)) {
    return { safe: false, reason: "path escapes repository root" };
  }
  return { safe: true, resolved };
}

// Escape literal `{`/`}` outside code and math so MDX doesn't parse them as JS
// expressions. LaTeX groups must remain intact for remark-math and KaTeX.
function escapeMdxBraces(text) {
  const fenceRegex = /(```[\s\S]*?```)/g;
  return text
    .split(fenceRegex)
    .map((segment, i) => {
      if (i % 2 === 1) return segment;
      const protectedInlineRegex = /(\$\$[\s\S]*?\$\$|\$[^$\n]+\$|`[^`]*`)/g;
      return segment
        .split(protectedInlineRegex)
        .map((sub, j) => {
          if (j % 2 === 1) return sub;
          return sub.replace(/(?<!\\)([{}])/g, "\\$1");
        })
        .join("");
    })
    .join("");
}

async function main() {
  const logbookRaw = await readText(LOGBOOK_PATH);

  const decisions = parseDecisions(logbookRaw);

  const imageRegex = /!\[([^\]]*)\]\(([^)\s]+)\)/g;
  const copyOps = [];
  let imageCount = 0;
  let rewritten = logbookRaw.replace(imageRegex, (match, alt, rawPath) => {
    const verdict = classifyImagePath(rawPath);
    if (!verdict.safe) {
      throw new Error(`Unsafe Logbook image path "${rawPath}": ${verdict.reason}`);
    }
    copyOps.push({ rawPath, resolved: verdict.resolved });
    const newPath = `/generated/logbook-assets/${rawPath}`;
    return `![${alt}](${newPath})`;
  });

  const existingCopyOps = [];
  const missingImages = [];
  for (const op of copyOps) {
    try {
      await fs.access(op.resolved);
      existingCopyOps.push(op);
      imageCount += 1;
    } catch {
      missingImages.push(op.rawPath);
    }
  }

  if (missingImages.length > 0) {
    throw new Error(`Missing Logbook image assets:\n${missingImages.join("\n")}`);
  }

  rewritten = escapeMdxBraces(rewritten);

  // Replace generated output deterministically.
  await fs.rm(SRC_GENERATED, { recursive: true, force: true });
  await fs.rm(PUBLIC_GENERATED, { recursive: true, force: true });
  await fs.mkdir(SRC_GENERATED, { recursive: true });
  await fs.mkdir(LOGBOOK_ASSETS_DIR, { recursive: true });

  const imageDimensions = {};
  for (const op of existingCopyOps) {
    const dest = path.join(LOGBOOK_ASSETS_DIR, op.rawPath);
    await fs.mkdir(path.dirname(dest), { recursive: true });
    await fs.copyFile(op.resolved, dest);
    const publicPath = `/generated/logbook-assets/${op.rawPath}`;
    imageDimensions[publicPath] = readPngDimensions(
      await fs.readFile(op.resolved),
      op.rawPath,
    );
  }

  await fs.writeFile(path.join(SRC_GENERATED, "logbook.mdx"), rewritten, "utf8");

  const sourceChecksum = `sha256:${sha256(Buffer.from(logbookRaw, "utf8"))}`;

  const logbookIndex = {
    sourceChecksum,
    decisionCount: decisions.length,
    imageCount,
    imageDimensions,
    decisions,
  };
  await fs.writeFile(
    path.join(SRC_GENERATED, "logbook-index.json"),
    JSON.stringify(logbookIndex, null, 2) + "\n",
    "utf8",
  );

  await fs.copyFile(PGN_PATH, path.join(PUBLIC_GENERATED, "official-elo-clean.pgn"));

  const ratingsRaw = await readText(RATINGS_PATH);
  const leaderboard = parseRatings(ratingsRaw);
  await fs.writeFile(
    path.join(SRC_GENERATED, "leaderboard.json"),
    JSON.stringify(leaderboard, null, 2) + "\n",
    "utf8",
  );

  console.log(
    `generate-content: wrote ${decisions.length} decisions, ${imageCount} images, ${leaderboard.length} leaderboard rows`,
  );
}

function parseRatings(text) {
  const rowRegex =
    /^\s*(\d+)\s+(\S+)\s*:\s+([\d.]+)\s+(-+|[\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s*$/;
  const rows = [];
  for (const line of text.split("\n")) {
    const match = line.match(rowRegex);
    if (!match) continue;
    const [, rank, player, rating, error, points, played, percent] = match;
    rows.push({
      rank: Number(rank),
      player,
      rating: Number(rating),
      error: /^-+$/.test(error) ? null : Number(error),
      points: Number(points),
      played: Number(played),
      percent: Number(percent),
    });
  }
  return rows;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
