import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 1. Load lightweight model (fast inference, good semantic quality)
model = SentenceTransformer("all-MiniLM-L6-v2")

catalog_path = Path("data/catalog.jsonl")
output_index_path = Path("data/catalog_embeddings.npy")
output_ids_path = Path("data/catalog_asins.json")

print("Loading catalog...")
texts = []
asins = []

with open(catalog_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        asin = item.get("asin") or item.get("item_id")

        # Combine rich fields for semantic indexing: title, brand, category, description
        title = item.get("title", "")
        brand = item.get("brand", "")
        categories = " ".join(item.get("categories", []))
        desc = item.get("description", "")
        if isinstance(desc, list):
            desc = " ".join(desc)

        doc_text = f"Title: {title} | Brand: {brand} | Categories: {categories} | Details: {desc[:200]}"

        asins.append(asin)
        texts.append(doc_text)

print(f"Loaded {len(texts)} products. Generating embeddings...")

# 2. Compute normalized embeddings (so dot product == cosine similarity)
embeddings = model.encode(
    texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
)

# 3. Save to disk for fast startup in the agent
np.save(output_index_path, embeddings.astype(np.float32))
with open(output_ids_path, "w", encoding="utf-8") as f:
    json.dump(asins, f)

print(f"Saved embeddings to {output_index_path}")
