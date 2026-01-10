import "dotenv/config";
import fs from "node:fs/promises";
import path from "node:path";
import OpenAI from "openai";

export const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export function normalizeText(s) {
  return s
    .replace(/\r\n/g, "\n")
    .replace(/[ \t]+/g, " ")
    .trim();
}

export function chunkText(text, { maxChars = 1200, overlapChars = 200 } = {}) {
  // Simple, robust chunker (character-based).
  // For production, switch to token-based chunking.
  const t = normalizeText(text);
  const chunks = [];
  let i = 0;

  while (i < t.length) {
    const end = Math.min(i + maxChars, t.length);
    const chunk = t.slice(i, end);
    chunks.push(chunk);
    if (end === t.length) break;
    i = Math.max(0, end - overlapChars);
  }
  return chunks;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function l2norm(v) {
  return Math.sqrt(dot(v, v));
}

export function normalizeVec(v) {
  const n = l2norm(v) || 1;
  return v.map((x) => x / n);
}

export function cosineSimUnitVectors(aUnit, bUnit) {
  // If vectors are already normalized, cosine = dot product
  return dot(aUnit, bUnit);
}

export async function saveJSON(filePath, obj) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify(obj, null, 2), "utf-8");
}

export async function loadJSON(filePath) {
  const raw = await fs.readFile(filePath, "utf-8");
  return JSON.parse(raw);
}
