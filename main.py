import json
import pandas as pd
import os
import re

# -------------------------
# Input/Output Config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARXIV_IN = "arxiv-metadata-oai-snapshot.json"
RW_IN = "retraction-watch-database.csv"
ARXIV_OUT = os.path.join(BASE_DIR, "arxiv_cs.json")
RW_OUT = os.path.join(BASE_DIR, "retraction_watch_cs.csv")

# Constants
MAX_ARXIV_SAMPLES = 100_000
CS_ARXIV_CATS = {"cs.AI", "cs.LG", "cs.CL", "cs.SE", "cs.DB", "cs.IR", "cs.CR", "cs.DS", "cs.NI"}
CS_RW_KEYWORDS = ["computer", "software", "information system", "artificial intelligence", "machine learning",
                  "data science", "cyber", "network", "database"]


# -------------------------
# Helper functions
# -------------------------
def clean_paper_title(title):
    """Removes 'RETRACTED' prefixes that cause data leakage."""
    # Remove common retraction prefixes (case-insensitive)
    prefixes = [r"^retracted article:", r"^retracted:", r"^retraction:", r"^withdrawn:"]
    cleaned = title.strip()
    for p in prefixes:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def get_arxiv_year(arxiv_id):
    """Extracts year from arXiv ID (Format: YYMM.NNNNN)."""
    try:
        # Modern IDs start with YYMM
        year_prefix = arxiv_id.split('.')[0][:2]
        return int("20" + year_prefix)
    except:
        return None


def is_cs_arxiv(categories):
    if not isinstance(categories, str): return False
    return any(cat in categories.split() for cat in CS_ARXIV_CATS)


def is_cs_rw(subject):
    if not isinstance(subject, str): return False
    return any(k in subject.lower() for k in CS_RW_KEYWORDS)

def build_text(title, abstract, max_words=200):
    """
    Combine title and abstract for downstream semantic embedding.
    Abstract is truncated to avoid length bias.
    """
    abstract_words = abstract.split()
    abstract = " ".join(abstract_words[:max_words])
    return f"{title} [SEP] {abstract}"

# -------------------------
# 1. Process Retraction Watch First
# -------------------------
print("Processing Retraction Watch...")
rw = pd.read_csv(RW_IN, encoding='latin1')  # Some RW files use latin1
rw_cs = rw[rw["Subject"].apply(is_cs_rw)].copy()

# Clean titles and extract date range
rw_cs['Title'] = rw_cs['Title'].apply(clean_paper_title)
rw_cs['Year'] = pd.to_datetime(rw_cs['OriginalPaperDate']).dt.year
valid_years = rw_cs['Year'].dropna().unique().astype(int).tolist()

rw_cs.to_csv(RW_OUT, index=False)
print(f"Retracted papers found for years: {min(valid_years)} to {max(valid_years)}")

# -------------------------
# 2. Filter arXiv to Match
# -------------------------
print(f"\nFiltering arXiv (Matching years {min(valid_years)}-{max(valid_years)})...")

n_read, n_kept = 0, 0
with open(ARXIV_IN, "r", encoding="utf-8") as fin, \
        open(ARXIV_OUT, "w", encoding="utf-8") as fout:
    for line in fin:
        if n_kept >= MAX_ARXIV_SAMPLES:  # Limit saved papers, not just read lines
            break
        n_read += 1

        try:
            paper = json.loads(line)
        except:
            continue

        # Filter 1: Category
        if not is_cs_arxiv(paper.get("categories", "")):
            continue

        # Filter 2: Year Alignment
        paper_year = get_arxiv_year(paper.get("id", ""))
        if paper_year not in valid_years:
            continue

        # Filter 3: Quality and Cleaning
        title = clean_paper_title(paper.get("title", "").replace("\n", " "))
        abstract = paper.get("abstract", "").replace("\n", " ").strip()

        if len(title.split()) < 5 or len(abstract.split()) < 50:
            continue

        paper['title'] = title
        paper['abstract'] = abstract
        paper['text'] = build_text(title, abstract)

        fout.write(json.dumps(paper) + "\n")
        n_kept += 1

print(f"[arXiv] Scanned {n_read:,} → Kept {n_kept:,} matched CS papers.")
