"""
Matching Engine Demo (PM Portfolio)

Purpose:
- Demonstrate how a platform might score 'talent' vs 'project needs'
- Uses TF-IDF + cosine similarity as a lightweight proxy for embeddings
- Outputs ranked candidates + a simple explanation

Details
-TF-IDF (Term Frequency-Inverse Document Frequency) is a weighting scheme that converts text into numerical vectors.
-Cosine Similarity is a metric that measures the angle between those vectors to determine how similar they are. 

Run:
  python matching_demo.py
"""

from dataclasses import dataclass
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Candidate:
    name: str
    profile: str


@dataclass
class Project:
    title: str
    description: str


def rank_candidates(project: Project, candidates: List[Candidate], top_k: int = 5) -> List[Tuple[Candidate, float]]:
    docs = [project.description] + [c.profile for c in candidates]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(docs)

    project_vec = X[0]
    cand_vecs = X[1:]
    scores = cosine_similarity(project_vec, cand_vecs).flatten()

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def main():
    project = Project(
        title="AI Project: Customer Churn Risk + Retention Triggers",
        description=(
            "Build an ML model to predict subscription churn and recommend retention triggers. "
            "Needs experience with cohort analysis, experimentation, data pipelines, and deploying models. "
            "Prefer prior work in DTC subscriptions or consumer products."
        ),
    )

    candidates = [
        Candidate(
            name="Asha (ML Engineer)",
            profile="Built churn prediction models using XGBoost and logistic regression. "
                    "Deployed models with batch pipelines, feature stores, and monitoring. "
                    "Worked on subscription retention and lifecycle triggers.",
        ),
        Candidate(
            name="Ben (Data Scientist)",
            profile="Strong experimentation and A/B testing background. Built dashboards, cohort analyses, "
                    "and causal inference studies. Some ML modeling but more analytics focused.",
        ),
        Candidate(
            name="Carla (Product Manager - Growth)",
            profile="Led subscription funnel optimization, retention programs, lifecycle messaging, and pricing tests. "
                    "Partnered with DS/Eng to ship churn models and automated CRM triggers.",
        ),
        Candidate(
            name="Diego (Backend Engineer)",
            profile="Built scalable APIs and microservices. Integrated payment systems and subscription billing. "
                    "Limited ML experience but strong platform engineering background.",
        ),
        Candidate(
            name="Eve (Applied Scientist)",
            profile="NLP and recommender systems. Experience with embeddings, semantic search, and retrieval systems. "
                    "Less direct subscription churn experience.",
        ),
    ]

    ranked = rank_candidates(project, candidates, top_k=5)

    print(f"\nProject: {project.title}\n")
    print("Top matches:")
    for idx, (cand, score) in enumerate(ranked, start=1):
        print(f"{idx}. {cand.name:25s}  score={score:.3f}")
        print(f"   profile: {cand.profile[:120]}{'...' if len(cand.profile) > 120 else ''}")
    print("\nNote: This demo uses TF-IDF similarity as a lightweight proxy for embeddings.\n")


if __name__ == "__main__":
    main()
