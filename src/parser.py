# src/parser.py
import os
import json
from pathlib import Path
import re

def parse_resume_text(text):
    """Extract sections like skills, experience, education."""
    text = text.lower()

    # Simple regex section extraction
    sections = {
        "skills": re.findall(r'skills?[:\-]\s*(.*)', text),
        "experience": re.findall(r'experience[:\-]\s*(.*)', text),
        "education": re.findall(r'education[:\-]\s*(.*)', text),
        "summary": re.findall(r'summary[:\-]\s*(.*)', text)
    }

    # Fallback: if empty, keep some text
    for key, val in sections.items():
        if not val:
            sections[key] = [text[:400]]  # take first few lines

    return sections


def main():
    input_dir = Path("data/raw_resumes_text")
    output_dir = Path("data/parsed_resumes")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.txt"):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            parsed = {
                "id": file.stem,
                "raw_text": text[:2000],  # short preview
                "sections": parse_resume_text(text),
                "meta": {"filename": file.name}
            }

            out_path = output_dir / f"{file.stem}.json"
            json.dump(parsed, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error parsing {file.name}: {e}")

    print(f"Parsing complete. Files saved to {output_dir}")


if __name__ == "__main__":
    main()
