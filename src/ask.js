/**
 * ask.js (Improvised / KT-friendly)
 * --------------------------------
 * What this file does (end-to-end):
 * 1) Reads a user question from CLI
 * 2) Loads a local vector index (chunks + embeddings) from disk
 * 3) "Augments" the query (Multi-query + HyDE) to improve retrieval recall
 * 4) Embeds the query variants (with caching)
 * 5) Retrieves the most relevant chunks (and de-duplicates / diversifies them)
 * 6) Sends only the selected context to the LLM to answer (with citations)
 *
 * Improvements in this version:
 * ✅ Better error handling: quota (insufficient_quota), rate limits, transient failures
 * ✅ Caching: embeddings cache + augmentation cache + answer cache (optional)
 * ✅ Cleaner logging with debug mode
 * ✅ KT-friendly structure + clear sectioning
 */

import fs from "node:fs/promises";
import path from "node:path";

import { client } from "./lib.js";
import { LocalVectorStore } from "./vectorStore.js";
import { embedTexts } from "./embed.js";
import {
  ANSWER_INSTRUCTIONS,
  MULTI_QUERY_INSTRUCTIONS,
  HYDE_INSTRUCTIONS,
} from "./prompts.js";

/* ----------------------------- Config ----------------------------- */

const INDEX_PATH = "index/store.json";

// Caches (simple JSON files on disk)
const CACHE_DIR = ".cache";
const EMBED_CACHE_PATH = path.join(CACHE_DIR, "embeddings.json");
const AUGMENT_CACHE_PATH = path.join(CACHE_DIR, "augment.json");
const ANSWER_CACHE_PATH = path.join(CACHE_DIR, "answers.json");

// Models (change if you want)
const GEN_MODEL = process.env.RAG_GEN_MODEL || "gpt-4.1-mini";
const EMBED_MODEL = process.env.RAG_EMBED_MODEL || "text-embedding-3-small";

// Retrieval / context tuning
const PER_QUERY_TOPK = Number(process.env.RAG_PER_QUERY_TOPK || 8);
const FINAL_TOPK = Number(process.env.RAG_FINAL_TOPK || 25);
const CONTEXT_K = Number(process.env.RAG_CONTEXT_K || 6);

// Augmentation toggles (you can disable to reduce cost / calls)
const ENABLE_MULTI_QUERY = (process.env.RAG_MULTI_QUERY ?? "true") === "true";
const ENABLE_HYDE = (process.env.RAG_HYDE ?? "true") === "true";

// Logging controls
const DEBUG = (process.env.RAG_DEBUG ?? "false") === "true";

/* ----------------------------- Logging ---------------------------- */

const log = {
  info: (msg, obj) =>
    console.log(`ℹ️  ${msg}${obj ? " " + JSON.stringify(obj) : ""}`),
  ok: (msg, obj) =>
    console.log(`✅ ${msg}${obj ? " " + JSON.stringify(obj) : ""}`),
  warn: (msg, obj) =>
    console.warn(`⚠️  ${msg}${obj ? " " + JSON.stringify(obj) : ""}`),
  err: (msg, obj) =>
    console.error(`❌ ${msg}${obj ? " " + JSON.stringify(obj) : ""}`),
  debug: (msg, obj) => {
    if (DEBUG) console.log(`🐛 ${msg}${obj ? " " + JSON.stringify(obj) : ""}`);
  },
};

/* ----------------------------- Helpers ---------------------------- */

/**
 * Very lightweight stable hashing for cache keys.
 * Not crypto; just deterministic for strings.
 */
function hashKey(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16);
}

async function ensureCacheDir() {
  await fs.mkdir(CACHE_DIR, { recursive: true });
}

async function readJsonSafe(filePath, fallbackObj) {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw);
  } catch {
    return fallbackObj;
  }
}

async function writeJson(filePath, obj) {
  await ensureCacheDir();
  await fs.writeFile(filePath, JSON.stringify(obj, null, 2), "utf-8");
}

/**
 * Identify OpenAI error types in a KT-friendly way.
 * - Quota: 429 + code=insufficient_quota
 * - Rate limit: 429 + code=rate_limit_exceeded (or message includes rate limit)
 * - Transient: 5xx, network timeouts, etc.
 */
function classifyOpenAIError(err) {
  const status = err?.status;
  const code = err?.code || err?.error?.code;
  const message = err?.message || err?.error?.message || "";

  const isQuota =
    status === 429 &&
    (code === "insufficient_quota" || message.toLowerCase().includes("quota"));

  const isRateLimit =
    status === 429 &&
    (code === "rate_limit_exceeded" ||
      message.toLowerCase().includes("rate limit") ||
      message.toLowerCase().includes("too many requests"));

  const isTransient =
    (status >= 500 && status <= 599) ||
    message.toLowerCase().includes("timeout") ||
    message.toLowerCase().includes("temporarily");

  return { status, code, message, isQuota, isRateLimit, isTransient };
}

/**
 * Retry wrapper with exponential backoff for rate limits / transient errors.
 * - Quota errors should NOT retry (won't help).
 */
async function withRetry(fn, { label = "operation", maxRetries = 4 } = {}) {
  let attempt = 0;

  while (true) {
    try {
      return await fn();
    } catch (err) {
      const info = classifyOpenAIError(err);

      // Do not retry quota errors
      if (info.isQuota) {
        throw err;
      }

      // Retry only for rate-limit or transient issues
      const canRetry = info.isRateLimit || info.isTransient;

      if (!canRetry || attempt >= maxRetries) {
        throw err;
      }

      const backoffMs =
        Math.min(8000, 500 * Math.pow(2, attempt)) +
        Math.floor(Math.random() * 250);
      log.warn(`${label} failed; retrying...`, {
        attempt: attempt + 1,
        backoffMs,
        status: info.status,
        code: info.code,
      });

      await new Promise((r) => setTimeout(r, backoffMs));
      attempt++;
    }
  }
}

/* ---------------------- Retrieval Diversity (MMR-ish) ---------------------- */

/**
 * pickDiverse()
 * - Input: hits sorted by relevance score (desc)
 * - Output: top-k hits that are not near-duplicates
 *
 * KT explanation:
 * We want "high relevance" AND "diversity", so we penalize chunks that are
 * too similar to already picked chunks (MMR concept).
 */
function pickDiverse(
  hits,
  { k = CONTEXT_K, lambda = 0.8, minKeep = 0.1 } = {}
) {
  const picked = [];
  const pickedEmbeds = [];

  for (const h of hits) {
    if (picked.length >= k) break;

    // similarity to picked chunks (cosine = dot since normalized vectors)
    let maxSimToPicked = 0;
    for (const pe of pickedEmbeds) {
      let sim = 0;
      const a = h.item.embeddingUnit;
      const b = pe;
      for (let i = 0; i < a.length; i++) sim += a[i] * b[i];
      if (sim > maxSimToPicked) maxSimToPicked = sim;
    }

    // MMR-ish score
    const mmrScore = lambda * h.score - (1 - lambda) * maxSimToPicked;

    // keep if not basically a duplicate
    if (picked.length === 0 || mmrScore > minKeep) {
      picked.push(h);
      pickedEmbeds.push(h.item.embeddingUnit);
    }
  }

  return picked;
}

/* ----------------------------- Augmentation ----------------------------- */

/**
 * Multi-query rewrite:
 * Ask LLM to produce 3 alternative search queries (JSON).
 * Cached by (model + question).
 */
async function getMultiQueriesCached(question, cache) {
  const key = `mq:${GEN_MODEL}:${hashKey(question)}`;
  if (cache[key]) {
    log.debug("Multi-query cache hit");
    return cache[key];
  }

  const resp = await withRetry(
    () =>
      client.responses.create({
        model: GEN_MODEL,
        instructions: MULTI_QUERY_INSTRUCTIONS,
        input: question,
        temperature: 0.2,
      }),
    { label: "multi-query" }
  );

  const raw = resp.output_text?.trim() || "";
  let queries = [];

  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.queries)) queries = parsed.queries.slice(0, 3);
  } catch {
    // If model returns invalid JSON, we fall back to no rewrites
    queries = [];
  }

  cache[key] = queries;
  return queries;
}

/**
 * HyDE:
 * Ask LLM for a short hypothetical answer to embed for retrieval.
 * Cached by (model + question).
 */
async function getHydeCached(question, cache) {
  const key = `hyde:${GEN_MODEL}:${hashKey(question)}`;
  if (cache[key]) {
    log.debug("HyDE cache hit");
    return cache[key];
  }

  const resp = await withRetry(
    () =>
      client.responses.create({
        model: GEN_MODEL,
        instructions: HYDE_INSTRUCTIONS,
        input: question,
        temperature: 0.3,
      }),
    { label: "hyde" }
  );

  const hyde = (resp.output_text || "").trim();
  cache[key] = hyde;
  return hyde;
}

/* ----------------------------- Embedding Cache ----------------------------- */

/**
 * Embedding cache stores embeddings per text+model.
 * NOTE: This can get large. For production, use:
 * - LRU cache
 * - SQLite
 * - Redis
 */
async function embedTextsCached(texts, embedCache) {
  // Build keys and find misses
  const keys = texts.map((t) => `emb:${EMBED_MODEL}:${hashKey(t)}`);
  const misses = [];
  const missIndexes = [];

  keys.forEach((k, i) => {
    if (!embedCache[k]) {
      misses.push(texts[i]);
      missIndexes.push(i);
    }
  });

  // Fetch embeddings for misses
  if (misses.length > 0) {
    log.debug("Embedding cache misses", { count: misses.length });

    const vectors = await withRetry(
      () => embedTexts(misses, { model: EMBED_MODEL }),
      { label: "embeddings" }
    );

    // Store them
    vectors.forEach((v, j) => {
      const originalIndex = missIndexes[j];
      embedCache[keys[originalIndex]] = v;
    });
  } else {
    log.debug("All embeddings served from cache");
  }

  // Return in original order
  return keys.map((k) => embedCache[k]);
}

/* ----------------------------- Context Builder ----------------------------- */

function buildContextBlock(selectedHits) {
  return selectedHits
    .map((h) => {
      const { source, chunkIndex, content } = h.item;
      return `[source: ${source}#${chunkIndex}]\n${content}`;
    })
    .join("\n\n---\n\n");
}

/* ----------------------------- Answering ----------------------------- */

async function answerWithContextCached(question, context, answerCache) {
  // Cache by (model + question + context hash).
  // If documents change, context hash changes, so cache invalidates naturally.
  const key = `ans:${GEN_MODEL}:${hashKey(question)}:${hashKey(context)}`;

  if (answerCache[key]) {
    log.debug("Answer cache hit");
    return answerCache[key];
  }

  const resp = await withRetry(
    () =>
      client.responses.create({
        model: GEN_MODEL,
        instructions: ANSWER_INSTRUCTIONS,
        input: `CONTEXT:\n\n${context}\n\nUSER QUESTION:\n${question}`,
        temperature: 0.2,
      }),
    { label: "answer" }
  );

  const out = resp.output_text;
  answerCache[key] = out;
  return out;
}

/* ----------------------------- Main Flow ----------------------------- */

async function main() {
  const question = process.argv.slice(2).join(" ").trim();
  if (!question) {
    console.log(`Usage:\n  npm run ask -- "Your question here"`);
    process.exit(1);
  }

  // 1) Load caches
  const embedCache = await readJsonSafe(EMBED_CACHE_PATH, {});
  const augmentCache = await readJsonSafe(AUGMENT_CACHE_PATH, {});
  const answerCache = await readJsonSafe(ANSWER_CACHE_PATH, {});

  // 2) Load vector store
  log.info("Loading vector index...", { INDEX_PATH });
  const store = await LocalVectorStore.load(INDEX_PATH);
  log.ok("Index loaded", { chunks: store.items.length });

  // 3) Augment query (graceful fallback on quota)
  let rewrites = [];
  let hyde = "";

  try {
    if (ENABLE_MULTI_QUERY)
      rewrites = await getMultiQueriesCached(question, augmentCache);
    if (ENABLE_HYDE) hyde = await getHydeCached(question, augmentCache);
  } catch (err) {
    const info = classifyOpenAIError(err);

    if (info.isQuota) {
      // If no quota, we can still do basic RAG if embeddings + answering also require API.
      // For now: degrade to "no augmentation" and continue.
      log.warn(
        "No API quota for augmentation; continuing without rewrites/HyDE.",
        {
          code: info.code,
        }
      );
      rewrites = [];
      hyde = "";
    } else {
      throw err;
    }
  }

  const variantTexts = [question, ...rewrites, hyde].filter(Boolean);

  log.debug("Variant texts for retrieval", {
    count: variantTexts.length,
    rewritesCount: rewrites.length,
    hydeUsed: Boolean(hyde),
  });

  // 4) Embed variants (with caching)
  let variantEmbeds;
  try {
    variantEmbeds = await embedTextsCached(variantTexts, embedCache);
  } catch (err) {
    const info = classifyOpenAIError(err);
    if (info.isQuota) {
      log.err(
        "No API quota for embeddings. You must add credits or use local embeddings.",
        {
          code: info.code,
        }
      );
      process.exit(1);
    }
    throw err;
  }

  // 5) Retrieve: multi-embedding union + score merge
  const mergedHits = store.searchMulti(variantEmbeds, {
    perQueryTopK: PER_QUERY_TOPK,
    finalTopK: FINAL_TOPK,
  });

  // 6) Select diverse top chunks
  const selected = pickDiverse(mergedHits, { k: CONTEXT_K, lambda: 0.8 });

  // 7) Build context
  const context = buildContextBlock(selected);

  // 8) Answer (with caching)
  let answer;
  try {
    answer = await answerWithContextCached(question, context, answerCache);
  } catch (err) {
    const info = classifyOpenAIError(err);
    if (info.isQuota) {
      log.err(
        "No API quota for answering. Add credits or switch to local LLM.",
        {
          code: info.code,
        }
      );
      process.exit(1);
    }
    throw err;
  }

  // 9) Output
  console.log("\n===== ANSWER =====\n");
  console.log(answer);

  if (DEBUG) {
    console.log("\n===== DEBUG (RAG) =====\n");
    console.log("Rewrites:", rewrites);
    console.log(
      "HyDE:",
      hyde ? hyde.slice(0, 200) + (hyde.length > 200 ? "..." : "") : ""
    );
    console.log(
      "Selected sources:",
      selected.map((h) => h.item.id)
    );
    console.log("Config:", {
      GEN_MODEL,
      EMBED_MODEL,
      PER_QUERY_TOPK,
      FINAL_TOPK,
      CONTEXT_K,
      ENABLE_MULTI_QUERY,
      ENABLE_HYDE,
    });
  } else {
    // Cleaner summary logging (KT-friendly)
    log.ok("Done", {
      retrievedCandidates: mergedHits.length,
      contextChunks: selected.length,
      rewrites: rewrites.length,
      hyde: Boolean(hyde),
    });
  }

  // 10) Persist caches (important!)
  await writeJson(EMBED_CACHE_PATH, embedCache);
  await writeJson(AUGMENT_CACHE_PATH, augmentCache);
  await writeJson(ANSWER_CACHE_PATH, answerCache);

  log.debug("Caches saved");
}

main().catch((e) => {
  // Final safety net
  const info = classifyOpenAIError(e);

  if (info.isQuota) {
    log.err(
      "OpenAI quota exceeded / not enabled for this project. Add credits in billing.",
      {
        status: info.status,
        code: info.code,
      }
    );
    process.exit(1);
  }

  if (info.isRateLimit) {
    log.err(
      "Rate limited even after retries. Try again later or reduce calls.",
      {
        status: info.status,
        code: info.code,
      }
    );
    process.exit(1);
  }

  log.err("Unexpected error", { message: info.message || String(e) });
  if (DEBUG) console.error(e);
  process.exit(1);
});
