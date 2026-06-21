import io
import re
from collections import Counter

from docx import Document
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB total request


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.blank("en")

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")


# 🔹 RUN APP (VERY IMPORTANT FOR RENDER)





def extract_text_pdf(stream) -> str:
    data = stream.read()
    stream.seek(0)
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def extract_text_docx(stream) -> str:
    data = stream.read()
    stream.seek(0)
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def extract_resume_text(file_storage) -> str:
    name = (file_storage.filename or "").lower()
    if name.endswith(".pdf"):
        return extract_text_pdf(file_storage.stream)
    if name.endswith(".docx"):
        return extract_text_docx(file_storage.stream)
    return ""


def normalize_doc(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    return text


def ats_score(raw_text: str) -> int:
    if not raw_text or len(raw_text.strip()) < 50:
        return 0
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    text_lower = raw_text.lower()
    checks = 0
    total = 7

    if EMAIL_RE.search(raw_text):
        checks += 1
    if PHONE_RE.search(raw_text):
        checks += 1
    if len(raw_text.split()) >= 120:
        checks += 1
    if any(ln.startswith(("-", "•", "*", "·")) for ln in lines[:80]):
        checks += 1
    section_hits = sum(
        1
        for kw in (
            "experience",
            "education",
            "skills",
            "summary",
            "objective",
            "projects",
            "work",
        )
        if kw in text_lower
    )
    if section_hits >= 2:
        checks += 1
    if 5 <= len(lines) <= 200:
        checks += 1
    if not re.search(r"[^\x00-\x7F]", raw_text) or len(raw_text) < 8000:
        checks += 1

    return int(round(100 * checks / total))


def tfidf_similarity_percent(a: str, b: str) -> float:
    a, b = normalize_doc(a), normalize_doc(b)
    if len(a) < 20 or len(b) < 20:
        return 0.0
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    try:
        m = vec.fit_transform([a, b])
        sim = cosine_similarity(m[0:1], m[1:2])[0][0]
    except ValueError:
        return 0.0
    return float(max(0.0, min(1.0, sim)) * 100)


def keyword_scores(job_text: str, resume_text: str, top_n: int = 40):
    jd = normalize_doc(job_text)
    rs = normalize_doc(resume_text)
    if len(jd) < 30 or len(rs) < 30:
        return 0.0, [], []

    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    try:
        jd_mat = vec.fit_transform([jd])
        feats = vec.get_feature_names_out()
        scores = jd_mat.toarray()[0]
        order = scores.argsort()[::-1][:top_n]
        jd_terms = [feats[i] for i in order if scores[i] > 0]
    except ValueError:
        return 0.0, [], []

    if not jd_terms:
        return 0.0, [], []

    resume_set = set(rs.split())
    matched = [t for t in jd_terms if t in resume_set or t.replace(" ", "") in rs.replace(" ", "")]
    missing = [t for t in jd_terms if t not in matched][:25]
    matched = matched[:25]
    ratio = len(matched) / len(jd_terms) if jd_terms else 0.0
    return float(ratio * 100), matched, missing


def skill_relevance_percent(job_text: str, resume_text: str) -> float:
    doc_j = nlp((job_text or "")[:800000])
    doc_r = nlp((resume_text or "")[:800000])
    j_lemmas = {
        t.lemma_.lower()
        for t in doc_j
        if t.pos_ in ("NOUN", "PROPN") and not t.is_stop and t.is_alpha and len(t.text) > 2
    }
    r_lemmas = {
        t.lemma_.lower()
        for t in doc_r
        if t.pos_ in ("NOUN", "PROPN") and not t.is_stop and t.is_alpha and len(t.text) > 2
    }
    if not j_lemmas or not r_lemmas:
        return 0.0
    inter = len(j_lemmas & r_lemmas)
    return float(min(100.0, 100.0 * inter / max(1, len(j_lemmas))))


def top_resume_keywords(resume_text: str, k: int = 18):
    doc = nlp((resume_text or "")[:800000])
    words = [
        t.lemma_.lower()
        for t in doc
        if t.pos_ in ("NOUN", "PROPN") and not t.is_stop and t.is_alpha and len(t.text) > 2
    ]
    counts = Counter(words)
    return counts.most_common(k)


def build_suggestions(
    matched_keywords,
    missing_keywords,
    ats,
    keyword_pct,
    similarity_pct,
) -> list[str]:
    out = []
    if missing_keywords:
        out.append(
            "Weave missing job-description terms naturally into experience bullets: "
            + ", ".join(missing_keywords[:8])
            + ("…" if len(missing_keywords) > 8 else "")
        )
    if keyword_pct < 45:
        out.append(
            "Keyword coverage is low — mirror phrasing from the posting (tools, domains, metrics)."
        )
    if similarity_pct < 40:
        out.append(
            "Overall semantic overlap is weak — align your summary and skills with the role’s core themes."
        )
    if ats < 60:
        out.append(
            "ATS-style formatting: clear section headings, standard bullets, and contact details on the first page."
        )
    if not out:
        out.append("Strong alignment — tighten metrics and role-specific outcomes to stand out further.")
    return out
@app.route("/", methods=["GET", "POST"])
def index():
    job_description = ""
    errors = []
    results = []
    summary = None

    if request.method == "POST":
        job_description = (request.form.get("job_description") or "").strip()
        files = request.files.getlist("resumes")

        if not job_description:
            errors.append("Please paste a job description.")

        valid_files = [f for f in files if f and f.filename]
        if not valid_files:
            errors.append("Please upload at least one PDF or DOCX resume.")

        parsed = []

        for f in valid_files:
            safe = secure_filename(f.filename or "resume")
            ext = safe.lower().rsplit(".", 1)

            if len(ext) < 2 or ext[1] not in ("pdf", "docx"):
                errors.append(f"Skipped unsupported file: {safe}")
                continue

            text = extract_resume_text(f)

            if not text or len(text.strip()) < 30:
                errors.append(f"Could not read enough text from: {safe}")
                continue

            parsed.append((safe, text))

        if parsed and job_description:
            rows = []

            for name, resume_text in parsed:
                sim = tfidf_similarity_percent(job_description, resume_text)
                ats = ats_score(resume_text)
                kw_pct, matched_kw, missing_kw = keyword_scores(job_description, resume_text)
                sk = skill_relevance_percent(job_description, resume_text)

                final = round(0.35 * sim + 0.25 * ats + 0.25 * kw_pct + 0.15 * sk)
                decision = "ACCEPT" if final >= 52 else "REJECT"

                rows.append({
                    "resume_name": name,
                    "resume_text": resume_text,
                    "similarity_pct": round(sim, 1),
                    "ats_score": ats,
                    "keyword_score": round(kw_pct, 1),
                    "skill_relevance": round(sk, 1),
                    "final_score": final,
                    "decision": decision,
                    "matched_keywords": matched_kw,
                    "missing_keywords": missing_kw,
                    "top_resume_keywords": top_resume_keywords(resume_text),
                    "suggestions": build_suggestions(
                        matched_kw, missing_kw, ats, kw_pct, sim
                    ),
                })

            rows.sort(key=lambda r: r["final_score"], reverse=True)

            for i, r in enumerate(rows, start=1):
                r["rank"] = i

            if rows:
                rows[0]["is_top_candidate"] = True
                for r in rows[1:]:
                    r["is_top_candidate"] = False

            accepted = sum(1 for r in rows if r["decision"] == "ACCEPT")

            summary = {
                "total_resumes": len(rows),
                "accepted": accepted,
                "rejected": len(rows) - accepted,
                "best_candidate": rows[0]["resume_name"] if rows else "—",
            }

            results = rows

    return render_template(
        "index.html",
        job_description=job_description,
        errors=errors,
        results=results,
        summary=summary,
    )
# 🔹 ALWAYS LAST PART OF FILE
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
