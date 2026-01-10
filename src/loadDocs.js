import fs from "node:fs/promises";
import path from "node:path";
import { glob } from "glob";
import { chunkText } from "./lib.js";

export async function loadAndChunkDocs({
  dataDir = "data",
  exts = ["txt", "md"],
  chunk = { maxChars: 1200, overlapChars: 200 },
} = {}) {
  const patterns = exts.map((e) => path.join(dataDir, `**/*.${e}`));
  const files = (await glob(patterns, { nodir: true })).sort();

  const chunks = [];
  for (const file of files) {
    const text = await fs.readFile(file, "utf-8");
    const parts = chunkText(text, chunk);
    parts.forEach((content, idx) => {
      chunks.push({
        id: `${file}#${idx}`,
        source: file,
        chunkIndex: idx,
        content,
      });
    });
  }

  return chunks;
}
