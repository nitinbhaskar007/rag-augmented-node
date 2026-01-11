import { client } from "./lib.js";
import { LocalVectorStore } from "./vectorStore.js";
import { embedTexts } from "./embed.js";
import {
  ANSWER_INSTRUCTIONS,
  MULTI_QUERY_INSTRUCTIONS,
  HYDE_INSTRUCTIONS,
} from "./prompts.js";

const INDEX_PATH = "index/store.json";

/**
 * PURPOSE:
 * pickDiverse() is a simple "diversity selector" (MMR-ish).
 *
 * Why needed?
 * - If your top 6 chunks are all from the same paragraph,
 *   your context becomes repetitive and wastes tokens.
 *
 * So this tries to pick:
 * - highly relevant chunks
 * - but not near-duplicates of already picked ones
 *
 * lambda controls tradeoff:
 * - higher lambda => prioritize relevance
 * - lower lambda => prioritize diversity
 */
function pickDiverse(hits, { k = 6, lambda = 0.8 } = {}) {
  const picked = [];
  const pickedEmbeds = [];

  for (const h of hits) {
    if (picked.length >= k) break;

    // compute similarity of this candidate chunk to already picked chunks
    let maxSimToPicked = 0;

    for (const pe of pickedEmbeds) {
      // cosine similarity = dot product since vectors are unit normalized
      let sim = 0;
      const a = h.item.embeddingUnit;
      const b = pe;

      for (let i = 0; i < a.length; i++) sim += a[i] * b[i];
      if (sim > maxSimToPicked) maxSimToPicked = sim;
    }

    /**
     * MMR-like score:
     * - reward relevance (h.score)
     * - penalize similarity to what's already chosen (maxSimToPicked)
     *
     * If a chunk is super similar to something we already selected,
     * this score drops.
     */
    const mmrScore = lambda * h.score - (1 - lambda) * maxSimToPicked;

    // keep if not basically duplicate
    if (picked.length === 0 || mmrScore > 0.1) {
      picked.push(h);
      pickedEmbeds.push(h.item.embeddingUnit);
    }
  }

  return picked;
}

/**
 * Multi-query augmentation step:
 * Ask the LLM: "Rewrite this question into 3 different search queries"
 *
 * This helps retrieval when:
 * - user question is vague
 * - synonyms are needed
 * - you want different angles
 */
async function getMultiQueries(question, { model = "gpt-4.1-mini" } = {}) {
  const resp = await client.responses.create({
    model,
    instructions: MULTI_QUERY_INSTRUCTIONS,
    input: question,
    temperature: 0.2,
  });

  const raw = resp.output_text?.trim() || "";

  // Expect JSON like: {"queries":["...","...","..."]}
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.queries)) return parsed.queries.slice(0, 3);
  } catch {
    // If model returns non-JSON, fallback to empty
  }

  return [];
}

/**
 * HyDE augmentation step (Hypothetical Document Embeddings)
 *
 * You ask the LLM to "write a short answer" (hypothetical),
 * then embed that hypothetical answer.
 *
 * Why it helps:
 * - if the user question is abstract, the HyDE answer contains concrete terms
 * - embedding the HyDE text can retrieve better matching chunks
 */
async function getHydeText(question, { model = "gpt-4.1-mini" } = {}) {
  const resp = await client.responses.create({
    model,
    instructions: HYDE_INSTRUCTIONS,
    input: question,
    temperature: 0.3,
  });

  return (resp.output_text || "").trim();
}

/**
 * Build the context string that will be fed to the final answering model.
 * Includes source tags so the model can cite.
 */
function buildContextBlock(selectedHits) {
  return selectedHits
    .map((h) => {
      const { source, chunkIndex, content } = h.item;
      return `[source: ${source}#${chunkIndex}]\n${content}`;
    })
    .join("\n\n---\n\n");
}

/**
 * Final answer step:
 * We ask the LLM to answer ONLY from context.
 * If context doesn't contain answer => "I don't know..."
 */
async function answerWithContext(
  question,
  context,
  { model = "gpt-4.1-mini" } = {}
) {
  const resp = await client.responses.create({
    model,
    instructions: ANSWER_INSTRUCTIONS,
    input: `CONTEXT:\n\n${context}\n\nUSER QUESTION:\n${question}`,
    temperature: 0.2,
  });

  return resp.output_text;
}

async function main() {
  // Read question from command line:
  // npm run ask -- "your question"
  const question = process.argv.slice(2).join(" ").trim();
  if (!question) {
    console.log(`Usage:\n  npm run ask -- "Your question here"`);
    process.exit(1);
  }

  // Load our prebuilt index from disk (vectors + chunks)
  const store = await LocalVectorStore.load(INDEX_PATH);

  /**
   * AUGMENTED RAG:
   * 1) Multi-query rewrite: produce 3 alternate search queries
   * 2) HyDE: produce a short hypothetical answer
   *
   * We'll embed ALL of these and retrieve across them.
   */
  const rewrites = await getMultiQueries(question);
  const hyde = await getHydeText(question);

  // Variants we will embed for retrieval:
  // - original question
  // - 3 rewrites
  // - HyDE pseudo-answer
  const variantTexts = [question, ...rewrites, hyde].filter(Boolean);

  // Create embeddings for all variants (one API call)
  const variantEmbeds = await embedTexts(variantTexts);

  /**
   * Retrieve candidates using multiple query embeddings.
   * This improves recall:
   * - rewrite #1 might match doc section A
   * - rewrite #2 might match doc section B
   * - HyDE might match doc section C
   *
   * searchMulti merges and keeps the best score per chunk.
   */
  const mergedHits = store.searchMulti(variantEmbeds, {
    perQueryTopK: 8,
    finalTopK: 25,
  });

  /**
   * Pick top chunks with diversity (avoid duplicates)
   */
  const selected = pickDiverse(mergedHits, { k: 6, lambda: 0.8 });

  /**
   * Build a compact context block with source labels
   */
  const context = buildContextBlock(selected);

  /**
   * Ask the model to answer from context only
   */
  const answer = await answerWithContext(question, context);

  console.log("\n===== ANSWER =====\n");
  console.log(answer);

  // Helpful debug output:
  console.log("\n===== DEBUG (RAG) =====\n");
  console.log("Query rewrites:", rewrites);
  console.log("HyDE:", hyde.slice(0, 200) + (hyde.length > 200 ? "..." : ""));
  console.log(
    "Selected sources:",
    selected.map((h) => h.item.id)
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
