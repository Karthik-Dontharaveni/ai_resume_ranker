# src/ui_streamlit.py
# Streamlit UI for AI Resume Ranker
# Ensures project root is on sys.path so `from src.ranker import rank` works reliably.

import sys
import pathlib

# --- ensure project root is in sys.path ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now safe to import project modules
from src.ranker import rank
import streamlit as st

# --- UI layout ---
st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("AI Resume Ranker")
st.markdown("Paste a Job Description and get the top matching resumes from the corpus.")

# Input widgets
jd_text = st.text_area("Enter Job Description", height=220, placeholder="Paste job description here...")
skills_text = st.text_input("Comma-separated key skills (optional)", value="sous chef, food safety, inventory management")
top_k = st.slider("Number of candidates to show", 5, 20, 10)

# Action button
if st.button("Rank Resumes"):
    if not jd_text.strip():
        st.warning("Please paste a job description first.")
    else:
        jd_skills = [s.strip() for s in skills_text.split(",") if s.strip()]
        with st.spinner("Ranking resumes..."):
            try:
                results = rank(jd_text, jd_skills=jd_skills, top_k=top_k)
            except Exception as e:
                st.error(f"Ranking failed: {e}")
                results = []

        if not results:
            st.info("No matches found (empty results). Check your JD text or try different skills.")
        else:
            st.success(f"Top {len(results)} matches")
            # Display results in a nice card-like format
            for r in results:
                rid = r.get("id")
                score = r.get("score", 0.0)
                sim = r.get("sim", 0.0)
                skill_score = r.get("skill_score", 0.0)
                years = r.get("years", 0)

                st.markdown(
                    f"### `{rid}`  \n"
                    f"- **Score:** {score:.3f}  \n"
                    f"- **Similarity:** {sim:.3f}  \n"
                    f"- **Skill Match:** {skill_score:.2f}  \n"
                    f"- **Years (estimated):** {years}"
                )

                # optional: show small preview of raw resume text (if exists)
                raw_path = PROJECT_ROOT / "data" / "raw_resumes_text" / f"{rid}.txt"
                if raw_path.exists():
                    try:
                        preview = raw_path.read_text(encoding="utf-8", errors="ignore")[:800].strip()
                        st.code(preview, language="text")
                    except Exception:
                        pass

                st.divider()