# 🧠 AI Resume Ranker

An end-to-end machine learning project that ranks resumes based on a given job description using embeddings and FAISS.

## 🚀 Features
- Parses resumes and extracts skills
- Embeds resumes with Sentence Transformers
- Ranks them against job descriptions
- Interactive Streamlit UI
- Optional FastAPI backend

## 🧩 Tech Stack
Python, FAISS, Sentence Transformers, FastAPI, Streamlit, NumPy

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run src/ui_streamlit.py
