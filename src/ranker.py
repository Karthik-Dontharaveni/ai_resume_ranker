# src/ranker.py
"""
Ranker for AI Resume Ranker (TF-IDF embeddings pipeline).

Usage:
    python src/ranker.py            # runs a built-in example
    Or import rank(jd_text, jd_skills, top_k) from Python for custom queries.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

EMB_PATH = Path("data/embeddings/embeddings.npz")
FEAT_DIR = Path("data/featurized")
FAISS_DIR = Path("data/faiss")   # kept for compatibility though we use numpy similarity here

# TF-IDF vectorizer settings (must match src/embed.py)
TF_MAX_FEATURES = 8192
TF_NGRAM = (1, 2)

def load_embeddings():
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMB_PATH}")
    data = np.load(EMB_PATH, allow_pickle=True)
    ids = list(data["ids"])
    embeds = data["embeds"].astype("float32")
    metas = list(data["metas"])
    return ids, embeds, metas

def load_texts_for_vectorizer(ids):
    """
    Return texts ordered according to ids list.
    We load raw_text + skills (from featurized jsons) to build vectorizer consistent with embeddings.
    """
    texts = []
    for _id in ids:
        p = FEAT_DIR / f"{_id}.json"
        if not p.exists():
            texts.append("")  # fallback
            continue
        j = json.load(open(p, encoding="utf-8"))
        raw = j.get("raw_text", "")
        skills = " ".join(j.get("features", {}).get("skills", []))
        texts.append((raw + " " + skills).strip())
    return texts

def build_vectorizer_and_query_vector(jd_text, ids, limit=None):
    """
    Fit a TF-IDF vectorizer on the corpus (same params as embed.py),
    and transform the query (jd_text) into the same feature space.
    """
    corpus_texts = load_texts_for_vectorizer(ids if limit is None else ids[:limit])
    vec = TfidfVectorizer(max_features=TF_MAX_FEATURES, ngram_range=TF_NGRAM)
    X = vec.fit_transform(corpus_texts)   # we don't need X here (embeds saved), but fitting ensures same vocab
    q = vec.transform([jd_text]).astype("float32")
    q_dense = q.toarray().reshape(-1)
    # normalize q
    norm = np.linalg.norm(q_dense)
    if norm > 0:
        q_dense = q_dense / norm
    return q_dense, vec

def cosine_similarities(q_vec, embeds, limit=None):
    """
    Compute cosine similarities between q_vec and each row in embeds.
    embeds expected to be float32 numpy array (n_docs, dim).
    If embeddings are not normalized, normalize them here.
    """
    # normalize embeddings (rows)
    if embeds.ndim != 2:
        raise ValueError("Embeddings matrix must be 2D")
    emb_norms = np.linalg.norm(embeds, axis=1)
    # protect from zero norms
    emb_norms[emb_norms == 0] = 1.0
    normalized_embeds = embeds / emb_norms[:, None]
    # q_vec assumed normalized already
    sims = normalized_embeds.dot(q_vec)
    if limit:
        sims = sims[:limit]
    return sims

def skill_match_score(jd_skills, resume_skills):
    if not jd_skills:
        return 0.0
    jdset = set([s.lower().strip() for s in jd_skills if s and isinstance(s, str)])
    rset = set([s.lower().strip() for s in resume_skills if s and isinstance(s, str)])
    if not jdset or not rset:
        return 0.0
    return len(jdset.intersection(rset)) / max(len(jdset), 1)

def rank(jd_text, jd_skills=None, top_k=20, limit=None):
    """
    Rank resumes for a given job description.
    - jd_text: full job description string
    - jd_skills: optional list of skill tokens (strings) to boost matching
    - top_k: number of top results to return
    - limit: optional integer to consider only first N documents (speed)
    Returns: list of result dicts sorted by score
    """
    jd_skills = jd_skills or []
    ids, embeds, metas = load_embeddings()
    if limit:
        ids = ids[:limit]
        embeds = embeds[:limit]

    # Build TF-IDF vectorizer on corpus and get query vector
    q_vec, _vec = build_vectorizer_and_query_vector(jd_text, ids, limit=limit)
    sims = cosine_similarities(q_vec, embeds, limit=limit)

    results = []
    for idx, sim in enumerate(sims):
        rid = ids[idx]
        meta = metas[idx] if idx < len(metas) else {}
        # Read featurized to get skills & years
        feat_path = FEAT_DIR / f"{rid}.json"
        if feat_path.exists():
            j = json.load(open(feat_path, encoding="utf-8"))
            rskills = j.get("features", {}).get("skills", [])
            years = j.get("features", {}).get("years_exp", 0) or 0
        else:
            rskills, years = [], 0

        sscore = skill_match_score(jd_skills, rskills)
        # final scoring blend: tune weights here
        final = 0.70 * float(sim) + 0.25 * sscore + 0.05 * min(years / 10.0, 1.0)
        results.append({
            "id": rid,
            "sim": float(sim),
            "skill_score": float(sscore),
            "years": int(years),
            "score": float(final)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    # Built-in quick test
    example_jd = ("Looking for a Senior Machine Learning Engineer with 3+ years experience "
                  "in PyTorch, deployment, and AWS.")
    print("Running built-in test JD:")
    res = rank(example_jd, jd_skills=["pytorch", "aws", "docker"], top_k=10, limit=None)
    from pprint import pprint
    pprint(res[:10])