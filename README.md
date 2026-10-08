# Three-Phase Information Retrieval Engine

**Timeline:** May 2025  
**Category:** Main project

This engine was developed in three phases: document parsing and normalization, forward/inverted indexing with TF-IDF, and query processing with vector-space ranking and evaluation. The design supports reproducible indexing, cosine-similarity retrieval, and precision/recall analysis on TREC-style text collections.

## Stack

Python, NLTK, regular expressions, Porter stemming, forward indexing, inverted indexing, TF-IDF, vector-space model, cosine similarity, TREC, precision, and recall.

## Repository contents

- `src/` — parser, indexer, and query processor.
- `reports/` — phase reports.
- `data/README.md` — corpus provenance and excluded-data instructions.
