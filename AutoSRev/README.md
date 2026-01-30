# AutoSRev: Smart Outcome Summarizer for Domain-Specific Research Papers

## Authors and Creators
Brad Shea, Tyler Hollinger, Sarah Gothard

---

## Project Overview

**AutoSRev** is a prototype system that supports researchers in writing systematic reviews by automating parts of the literature filtering and data summarization process. Given a focused domain-specific query—such as *“late-stage multiple sclerosis treatments”*—AutoSRev retrieves a set of relevant scholarly abstracts, filters them for inclusion, and extracts outcome sentences that reflect each paper’s major findings and data spread.

Unlike basic literature tools, AutoSRev uses machine learning–based text classification to determine whether a paper fits the research scope and is worth summarizing. It then identifies and compiles key outcomes into a structured format for further synthesis.

---

## Use Case & Significance

Systematic reviews are essential in evidence-based medicine but are labor-intensive and time-consuming. Researchers must sift through hundreds of papers to find relevant studies and manually extract key results. AutoSRev accelerates this process by automating:
- Abstract filtering using a trained classifier
- Outcome sentence extraction using pattern-based NLP

This tool is especially useful for biomedical researchers, policy teams, and graduate students who need to rapidly screen literature and synthesize evidence in specific domains.

---

## How AutoSRev Differs from Existing Tools

Existing tools like Elicit.org and Google Scholar assist in citation discovery but do not offer:
- Domain-specific filtering using trainable models
- Automatic extraction of summary outcomes for publication use

AutoSRev fills this gap by combining vector similarity, classification, and outcome summarization into one streamlined workflow.

---

## System Architecture & Pipeline

### Languages & Libraries
- Python
- NLTK / spaCy (NLP)
- scikit-learn (vectorization and classification)


### Pipeline Steps

1. **Query & Retrieval**
   - User provides domain-specific query
   - Query expanded using Boolean keywords
   - Abstracts represented using TF-IDF vectors
   - Cosine similarity used to rank top-matching abstracts

2. **Text Preprocessing**
   - Lowercasing, tokenization, stopword removal, stemming
   - TF-IDF matrix creation

3. **Document Filtering**
   - Naive Bayes classifier trained on ~50–100 labeled abstracts
   - Compare performance of similarity-only vs classifier-enhanced retrieval

4. **Outcome Extraction**
   - Pattern-based rules to extract key outcome sentences (e.g., “significantly improved,” “reduced mortality”)

5. **Visualization (Future)**
   - Interface to display ranked abstracts, classification results, and extracted outcomes

---

## Data Sources

- **Corpus Focus**: Colon cancer
- **Documents**: 545 PMC full-text articles
- **Query Strategy**: Focused on “colon cancer” in title or abstract using Entrez + OAI-PMH API
- **Exclusion Handling**: Articles that were corrections, retractions, or non-research were marked as non-viable

### ![Non-viable articles](non_viable_list.png)

---

## Data Summary

### Word Count
- Range: 17 to 16,809 words
- Median: 4,280
- Mean: ~4,413 (SD = 2,439)

![Word Count Distribution](word_dist.png)

---

### Sentence Count
- Range: 1 to 1,110 sentences
- Median: 179
- Mean: ~201 (SD = 125)

![Sentence Spread](sent_spread.png)

These stats reflect the natural variability in biomedical literature, which the AutoSRev system must account for in modeling.

---

## Journal Distribution

A wide range of journals are represented. The most common was *Scientific Reports* (29 articles, ~5.3%), highlighting the diversity of publication sources.

![Top Journals Chart](journal_chart.png)

---

## Token Exploration

We excluded high-frequency domain terms like *colon*, *cancer*, *colorectal*, *patient*, and *patients* to highlight more meaningful content patterns.

### Word Cloud
Generated using Python’s `wordcloud` package.

![Word Cloud](wordclourd.png)

### Top 10 Most Frequent Tokens
- `cell`, `tumor`, `cells`, `expression`, `analysis`, `using`, `fig`, `study`, `group`, `treatment`

![Most Frequent Words](most_freq_words.png)

---

## Evaluation Plan

### Document Ranking
- Run 2 focused queries and evaluate ranked output
- Uses precision-at-k and recall for performance assessment
- Include negative control queries for specificity check

### Classifier Performance
- Evaluate Naive Bayes accuracy, precision, and recall
- Uses a held-out labeled validation set

### Outcome Extraction Accuracy
- Manually assess correctness of extracted outcome sentences
- Report match rate with paper’s key contributions

---

## Future Directions
- Incorporate lemmatization for improved token accuracy
- Expand rule-based outcome patterns with NLP dependency parsing
- Build and deploy a lightweight interface
- Explore integrating additional disease domains beyond colon cancer for broader use

---

## License
None

---

## Contact
For questions or collaborations, contact the authors

