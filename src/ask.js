import { client } from "./lib.js";
import { LocalVectorStore } from "./vectorStore.js";
import { embedText, embedTexts } from "./embed.js";
import {
  ANSWER_INSTRUCTIONS,
  MULTI_QUERY_INSTRUCTIONS,
  HYDE_INSTRUCTIONS,
} from "./prompts.js";

const INDEX_PATH = "index/store.json";

function pickDiverse(hits, { k = 6, lambda = 0.8 } = {}) {
  // Simple diversity selection:
  // - hits are already sorted by relevance score desc
  // - we greedily pick items penalizing near-duplicates using embedding similarity
  const picked = [];
  const pickedEmbeds = [];

  for (const h of hits) {
    if (picked.length >= k) break;

    let maxSimToPicked = 0;
    for (const pe of pickedEmbeds) {
      // embeddings are unit vectors; cosine = dot
      let sim = 0;
      const a = h.item.embeddingUnit;
      const b = pe;
      for (let i = 0; i < a.length; i++) sim += a[i] * b[i];
      if (sim > maxSimToPicked) maxSimToPicked = sim;
    }

    const mmrScore = lambda * h.score - (1 - lambda) * maxSimToPicked;

    // keep if it isn't basically a duplicate
    if (picked.length === 0 || mmrScore > 0.1) {
      picked.push(h);
      pickedEmbeds.push(h.item.embeddingUnit);
    }
  }

  return picked;
}

async function getMultiQueries(question, { model = "gpt-4.1-mini" } = {}) {
  const resp = await client.responses.create({
    model,
    instructions: MULTI_QUERY_INSTRUCTIONS,
    input: question,
    temperature: 0.2,
  });

  // Response text is in output_text in quickstart examples :contentReference[oaicite:7]{index=7}
  const raw = resp.output_text?.trim() || "";
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.queries)) return parsed.queries.slice(0, 3);
  } catch {}
  // Fallback if parsing fails
  return [];
}

async function getHydeText(question, { model = "gpt-4.1-mini" } = {}) {
  const resp = await client.responses.create({
    model,
    instructions: HYDE_INSTRUCTIONS,
    input: question,
    temperature: 0.3,
  });
  return (resp.output_text || "").trim();
}

function buildContextBlock(selectedHits) {
  return selectedHits
    .map((h) => {
      const { source, chunkIndex, content } = h.item;
      return `[source: ${source}#${chunkIndex}]\n${content}`;
    })
    .join("\n\n---\n\n");
}

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
  const question = process.argv.slice(2).join(" ").trim();
  if (!question) {
    console.log(`Usage:\n  npm run ask -- "Your question here"`);
    process.exit(1);
  }

  const store = await LocalVectorStore.load(INDEX_PATH);

  // 1) AUGMENT RAG
  const rewrites = await getMultiQueries(question);
  const hyde = await getHydeText(question);

  const variantTexts = [question, ...rewrites, hyde].filter(Boolean);

  // 2) Embed variants
  const variantEmbeds = await embedTexts(variantTexts);

  // 3) Retrieve union
  const mergedHits = store.searchMulti(variantEmbeds, {
    perQueryTopK: 8,
    finalTopK: 25,
  });

  // 4) Pick diverse top chunks
  const selected = pickDiverse(mergedHits, { k: 6, lambda: 0.8 });
  const context = buildContextBlock(selected);

  // 5) Generate final answer
  const answer = await answerWithContext(question, context);
  console.log("\n===== ANSWER =====\n");
  console.log(answer);

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
