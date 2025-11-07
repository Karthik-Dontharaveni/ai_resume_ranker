# src/index_faiss.py
import json
import numpy as np
import faiss
from pathlib import Path

def main():
    EMB_PATH = Path("data/embeddings/embeddings.npz")
    OUT_DIR = Path("data/faiss")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EMB_PATH.exists():
        print("Embeddings file not found! Run embed.py first.")
        return

    print("Loading embeddings...")
    data = np.load(EMB_PATH, allow_pickle=True)
    ids = data["ids"]
    embeds = data["embeds"].astype("float32")

    print(f"Loaded {len(ids)} vectors with dimension {embeds.shape[1]}")

    # Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)
    print(f"Index built with {index.ntotal} vectors")

    # Save index and IDs
    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    json.dump(list(ids), open(OUT_DIR / "ids.json", "w", encoding="utf-8"))
    print(f"Index and IDs saved to {OUT_DIR}")

if __name__ == "__main__":
    main()