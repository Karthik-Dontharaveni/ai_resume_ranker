# src/embed.py  (TF-IDF fallback, offline-friendly)
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

FEAT_DIR = Path("data/featurized")
OUT_DIR = Path("data/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_texts(limit=None):
    files = sorted(FEAT_DIR.glob("*.json"))
    if limit:
        files = files[:limit]
    ids = []
    texts = []
    metas = []
    for f in files:
        j = json.load(open(f, encoding="utf-8"))
        full = j.get("raw_text","")
        # include short features text too
        feats = j.get("features",{})
        skills = " ".join(feats.get("skills",[]))
        texts.append(full + " " + skills)
        ids.append(f.stem)
        metas.append({"file": f.name, "n_skills": feats.get("n_skills",0)})
    return ids, texts, metas

def embed_all(limit=None):
    ids, texts, metas = load_texts(limit)
    print("TF-IDF on", len(texts), "documents")
    vec = TfidfVectorizer(max_features=8192, ngram_range=(1,2))
    X = vec.fit_transform(texts)           # sparse matrix
    X = X.astype("float32").toarray()      # convert to dense float32
    np.savez(OUT_DIR / "embeddings.npz", ids=ids, embeds=X, metas=metas)
    print("Saved TF-IDF embeddings to", (OUT_DIR / "embeddings.npz").absolute())

if __name__ == "__main__":
    # default: all documents
    embed_all()