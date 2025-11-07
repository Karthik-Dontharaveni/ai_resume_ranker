# src/featurize.py
import os
import json
import re
from pathlib import Path

def extract_skills(text, skills_vocab):
    """Find skills present in text from skills vocabulary."""
    text_lower = text.lower()
    found = []
    for skill in skills_vocab:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found.append(skill)
    return list(set(found))


def estimate_experience(text):
    """Very simple rule-based experience estimate (years)."""
    matches = re.findall(r'(\d+)\+?\s*(?:year|yr)', text.lower())
    if matches:
        nums = [int(m) for m in matches]
        return max(nums)
    return 0


def main():
    parsed_dir = Path("data/parsed_resumes")
    output_dir = Path("data/featurized")
    output_dir.mkdir(parents=True, exist_ok=True)

    # skills.txt should be present (basic vocabulary)
    skills_file = Path("skills.txt")
    if not skills_file.exists():
        default_skills = ["python", "java", "sql", "excel", "communication", "aws", "machine learning",
                          "data analysis", "tensorflow", "pytorch", "leadership"]
        skills_file.write_text("\n".join(default_skills))
        print("⚠️ skills.txt not found. Default skills created.")

    skills_vocab = [s.strip() for s in skills_file.read_text(encoding="utf-8").splitlines() if s.strip()]

    for file in parsed_dir.glob("*.json"):
        try:
            data = json.load(open(file, encoding="utf-8"))
            text = data.get("raw_text", "")
            sections = data.get("sections", {})

            all_text = text + " " + " ".join([" ".join(v) for v in sections.values()])

            feats = {
                "skills": extract_skills(all_text, skills_vocab),
                "years_exp": estimate_experience(all_text),
                "n_skills": len(set(extract_skills(all_text, skills_vocab))),
            }

            data["features"] = feats
            out_path = output_dir / file.name
            json.dump(data, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error featurizing {file.name}: {e}")

    print(f"Featurization complete. Files saved to {output_dir}")


if __name__ == "__main__":
    main()