'''
Generating Corpus - only needed once
'''

# -----------------------------
# config
#------------------------------
import os
import requests
import xml.etree.ElementTree as ET
import time
import pandas as pd
import numpy as np
import nltk 
import string
import matplotlib.pyplot as plt
import seaborn as sns
import re
%matplotlib inline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer # build tf-idf matrix
import csv

EMAIL = os.getenv("NCBI_EMAIL", "your_email@domain.edu")
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OAI_BASE = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
TERM = "colon cancer"
MAX_RESULTS = 500

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# XML Scrape fuctions w/API call
#------------------------------

def get_pmcids(term=TERM, retmax=MAX_RESULTS):
    """Search PMC for PMCIDs matching the term."""
    url = f"{BASE_URL}esearch.fcgi"
    params = {
        "db": "pmc",
        "term": f"{term}[Title/Abstract]",
        "retmax": retmax,
        "retmode": "xml",
        "email": EMAIL,
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.content)
    pmcids = [id_elem.text for id_elem in root.findall(".//IdList/Id")]
    print(f"Found {len(pmcids)} PMC articles for term '{term}' in Title/Abstract only")
    return pmcids


def fetch_full_text_xml(pmcid):
    """Fetch full-text XML from PMC for given PMCID."""
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:pubmedcentral.nih.gov:{pmcid}",
        "metadataPrefix": "pmc",
    }
    response = requests.get(OAI_BASE, params=params)

    if response.status_code == 200:
        try:
            # Parse XML and check for <error> tag
            root = ET.fromstring(response.content)
            if root.tag.endswith("OAI-PMH"):
                error_tag = root.find(".//{http://www.openarchives.org/OAI/2.0/}error")
                if error_tag is not None:
                    print(f"Skipped PMCID {pmcid}: {error_tag.attrib.get('code')} — {error_tag.text.strip()}")
                    return None
            return response.content
        except ET.ParseError:
            print(f"Skipped PMCID {pmcid}: failed to parse XML content")
            return None
    else:
        print(f"Failed to fetch PMCID {pmcid} (status {response.status_code})")
        return None


def save_xml_file(pmcid, xml_content, output_dir=OUTPUT_DIR):
    """Save XML content to a file named by PMCID (with 'PMC' prefix)."""
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    path = os.path.join(output_dir, f"{pmcid}.xml")
    with open(path, "wb") as f:
        f.write(xml_content)
    return path

def collect_full_text_corpus(term=TERM, target_docs=500, overpull=800, output_dir=OUTPUT_DIR):
    """Fetch exactly `target_docs` XMLs, skipping already-downloaded PMCIDs."""
    os.makedirs(output_dir, exist_ok=True)

    # Get already saved PMCIDs (with prefix)
    existing_files = {
        fname.replace(".xml", "") for fname in os.listdir(output_dir) if fname.endswith(".xml")
    }
    saved_files = {pmcid: os.path.join(output_dir, f"{pmcid}.xml") for pmcid in existing_files}
    saved_count = len(saved_files)

    if saved_count >= target_docs:
        print(f"{saved_count} documents already saved — nothing to do.")
        return saved_files

    needed = target_docs - saved_count
    print(f"Already have {saved_count}. Need {needed} more documents...")
    pmcids = get_pmcids(term=term, retmax=overpull)

    for i, pmcid in enumerate(pmcids):
        pmcid_prefixed = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"

        if pmcid_prefixed in existing_files:
            continue
        if len(saved_files) >= target_docs:
            break

        xml = fetch_full_text_xml(pmcid)
        if xml:
            file_path = save_xml_file(pmcid, xml, output_dir=output_dir)
            saved_files[pmcid_prefixed] = file_path
            print(f"[{len(saved_files)}/{target_docs}] Saved PMCID {pmcid_prefixed} to {file_path}")
        else:
            print(f"[{len(saved_files)}/{target_docs}] Skipped PMCID {pmcid} due to fetch failure")

        time.sleep(0.3)  # Respect NCBI rate limits

    print(f"Finished with {len(saved_files)} total documents.")
    return saved_files

# -----------------------------
# Saving scraped XML
#------------------------------

corpus = collect_full_text_corpus() no arguments used for first pass

corpus = collect_full_text_corpus(term=TERM, target_docs=545, overpull=800, output_dir=OUTPUT_DIR) #used for second pass for additional 45

# -----------------------------
# Meta Data Extraction
#------------------------------

nltk.download("punkt")

def extract_corpus_metadata_robust(xml_dir, log_path="corpus_labels.csv"):
    rows = []

    ns = {
        "jats": "https://jats.nlm.nih.gov/ns/archiving/1.3/",
        "oai": "http://www.openarchives.org/OAI/2.0/"
    }

    for fname in os.listdir(xml_dir):
        if not fname.endswith(".xml"):
            continue

        pmcid = fname.replace(".xml", "")
        fpath = os.path.join(xml_dir, fname)

        try:
            tree = ET.parse(fpath)
            root = tree.getroot()

            title = root.find(".//jats:article-title", ns)
            journal = root.find(".//jats:journal-title", ns)
            pub_year = root.find(".//jats:pub-date/jats:year", ns)

            body = root.find(".//jats:body", ns)
            abstract = root.find(".//jats:abstract", ns)
            text = ""

            if body is not None:
                text = ET.tostring(body, encoding="unicode", method="text")
            elif abstract is not None:
                text = ET.tostring(abstract, encoding="unicode", method="text")

            clean_text = re.sub(r"\s+", " ", text or "").strip()

            if clean_text:
                word_count = len(clean_text.split())
                sentence_count = len(nltk.sent_tokenize(clean_text))
            else:
                word_count = 0
                sentence_count = 0

            rows.append({
                "pmcid": pmcid,
                "file_path": fpath,
                "title": title.text.strip() if title is not None and title.text else "",
                "journal": journal.text.strip() if journal is not None and journal.text else "",
                "pub_year": pub_year.text.strip() if pub_year is not None and pub_year.text else "Unknown",
                "word_count": word_count,
                "sentence_count": sentence_count,
                "label_q1": "",
                "label_q2": "",
                "checking_website": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
                "legit_coloncancer_article": "",
                "note": ""
            })

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    fieldnames = [
        "pmcid", "file_path", "title", "journal", "pub_year",
        "word_count", "sentence_count", "label_q1", "label_q2",
        "checking_website", "legit_coloncancer_article", "note"
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metadata log written to {log_path} with {len(rows)} documents.")

xml_dir = r"C:\Users\miked\Desktop2\RIT Info Retriv and Text Mining\project\data"
log_path = os.path.join(xml_dir, "corpus_labels.csv")
extract_corpus_metadata_robust(xml_dir, log_path)

