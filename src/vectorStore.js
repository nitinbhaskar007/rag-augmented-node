import {
  loadJSON,
  saveJSON,
  normalizeVec,
  cosineSimUnitVectors,
} from "./lib.js";

export class LocalVectorStore {
  constructor(items = []) {
    // items: [{ id, source, chunkIndex, content, embeddingUnit }]
    this.items = items;
  }

  static async load(filePath) {
    const data = await loadJSON(filePath);
    return new LocalVectorStore(data.items || []);
  }

  async save(filePath) {
    await saveJSON(filePath, { items: this.items });
  }

  add(item) {
    this.items.push(item);
  }

  search(queryEmbeddingUnit, { topK = 8 } = {}) {
    const scored = this.items.map((it) => ({
      item: it,
      score: cosineSimUnitVectors(queryEmbeddingUnit, it.embeddingUnit),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  // For multi-query retrieval: keep best score per chunk across multiple query embeddings
  searchMulti(queryEmbeddingUnits, { perQueryTopK = 8, finalTopK = 20 } = {}) {
    const best = new Map(); // id -> { item, score }

    for (const q of queryEmbeddingUnits) {
      const hits = this.search(q, { topK: perQueryTopK });
      for (const h of hits) {
        const prev = best.get(h.item.id);
        if (!prev || h.score > prev.score) best.set(h.item.id, h);
      }
    }

    const merged = [...best.values()].sort((a, b) => b.score - a.score);
    return merged.slice(0, finalTopK);
  }
}

export function toUnitEmbedding(embeddingFloatArray) {
  return normalizeVec(embeddingFloatArray);
}
