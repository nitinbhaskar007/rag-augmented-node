import { client, normalizeText } from "./lib.js";
import { toUnitEmbedding } from "./vectorStore.js";

export async function embedTexts(
  texts,
  { model = "text-embedding-3-small" } = {}
) {
  // Official embeddings endpoint :contentReference[oaicite:4]{index=4}
  const input = texts.map((t) => normalizeText(t));
  const res = await client.embeddings.create({
    model,
    input,
    encoding_format: "float",
  });

  // res.data[i].embedding is float[] unless you choose base64 encoding_format :contentReference[oaicite:5]{index=5}
  return res.data.map((d) => toUnitEmbedding(d.embedding));
}

export async function embedText(text, opts) {
  const [v] = await embedTexts([text], opts);
  return v;
}
