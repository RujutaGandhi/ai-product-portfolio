# Matching Engine Demo (Platform PM Artifact)

## Problem
In expert marketplaces (e.g., AI consulting networks), matching is often manual:
- Project needs are unstructured
- Talent profiles are inconsistent
- Ops teams spend time reading + interpreting + shortlisting

## Hypothesis
A lightweight matching engine can reduce ops load and improve match quality by:
1) structuring inputs
2) ranking candidates using semantic similarity
3) capturing feedback loops to improve over time

## What this prototype does
- Takes a project description + candidate profiles (free text)
- Computes similarity scores (TF-IDF + cosine similarity)
- Outputs a ranked shortlist

> TF-IDF is used here as a simple proxy for embeddings. Next step would be true embeddings + vector search.

## Key metrics (what I'd measure in production)
- **Time to shortlist** (minutes saved per match)
- **Shortlist acceptance rate** (talent accepts intro)
- **Match quality score** (client rating / project success)
- **Ops throughput** (matches per ops person per week)

## Roadmap (how this becomes real)
1) Convert talent + project profiles into structured fields (skills, constraints, domain)
2) Replace TF-IDF with embeddings (semantic vectors) + a vector database
3) Add a human-in-the-loop review UI with reason codes
4) Create feedback loops (accepted/declined, success outcomes) to retrain ranking
