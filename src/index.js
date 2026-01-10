import { loadAndChunkDocs } from "./loadDocs.js";
import { LocalVectorStore } from "./vectorStore.js";
import { embedTexts } from "./embed.js";

const INDEX_PATH = "index/store.json";

async function main() {
  console.log("Loading + chunking docs...");
  const chunks = await loadAndChunkDocs({
    dataDir: "data",
    exts: ["txt", "md"],
    chunk: { maxChars: 1200, overlapChars: 200 },
  });

  console.log(`Chunks: ${chunks.length}`);
  const store = new LocalVectorStore([]);

  // Batch embeddings to reduce requests
  const BATCH = 64;

  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const vectors = await embedTexts(batch.map((c) => c.content));

    batch.forEach((c, idx) => {
      store.add({
        id: c.id,
        source: c.source,
        chunkIndex: c.chunkIndex,
        content: c.content,
        embeddingUnit: vectors[idx],
      });
    });

    console.log(
      `Embedded ${Math.min(i + BATCH, chunks.length)}/${chunks.length}`
    );
  }

  await store.save(INDEX_PATH);
  console.log(`✅ Saved index: ${INDEX_PATH}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
