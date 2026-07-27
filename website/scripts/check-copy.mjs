#!/usr/bin/env node
// Fails with a nonzero exit if any authored or generated text file under
// website/ contains an em dash character.

import { promises as fs } from "node:fs";
import path from "node:path";

const WEBSITE_ROOT = process.cwd();
const EM_DASH = String.fromCodePoint(0x2014);

const IGNORED_DIR_NAMES = new Set(["node_modules", "dist", ".git"]);
const IGNORED_FILE_NAMES = new Set(["package-lock.json"]);
const BINARY_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".ico",
  ".webp",
  ".woff",
  ".woff2",
  ".ttf",
  ".eot",
  ".pgn",
  ".pdf",
]);

async function collectFiles(dir, out) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.isDirectory()) {
      if (IGNORED_DIR_NAMES.has(entry.name)) continue;
      await collectFiles(path.join(dir, entry.name), out);
      continue;
    }
    if (!entry.isFile()) continue;
    if (IGNORED_FILE_NAMES.has(entry.name)) continue;
    if (BINARY_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) continue;
    out.push(path.join(dir, entry.name));
  }
}

async function main() {
  const files = [];
  await collectFiles(WEBSITE_ROOT, files);

  const violations = [];
  for (const file of files) {
    const content = await fs.readFile(file, "utf8");
    const lines = content.split("\n");
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes(EM_DASH)) {
        violations.push({ file: path.relative(WEBSITE_ROOT, file), line: i + 1 });
      }
    }
  }

  if (violations.length > 0) {
    console.error("check-copy: em dash character found:");
    for (const v of violations) {
      console.error(`  ${v.file}:${v.line}`);
    }
    process.exit(1);
  }

  console.log(`check-copy: scanned ${files.length} files, no em dash found`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
