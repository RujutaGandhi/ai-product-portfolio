"""
Matching Engine Demo (PM Portfolio)

Purpose:
- Demonstrate how a platform might score 'talent' vs 'project needs'
- Uses TF-IDF + cosine similarity as a lightweight proxy for embeddings
- Simulates a RAG-style pattern by:
  - Retrieving top candidates
  - Building an LLM prompt with project + candidate context
  - Generating a stubbed "LLM explanation" for each match

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
    """
    Retrieval step: represent project + candidates as vectors and compute similarity.
    Acts as a lightweight stand-in for embeddings + vector search.
    """
    docs = [project.description] + [c.profile for c in candidates]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(docs)

    project_vec = X[0]
    cand_vecs = X[1:]
    scores = cosine_similarity(project_vec, cand_vecs).flatten()

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def build_llm_prompt(project: Project, candidate: Candidate) -> str:
    """
    RAG-style prompt builder:
    In a real system, this would include retrieved docs (e.g., past projects, ratings).
    Here we just use project + candidate text.
    """
    prompt = f"""
You are an expert matching assistant for an AI consulting marketplace.

Project:
Title: {project.title}
Description: {project.description}

Candidate:
Name: {candidate.name}
Profile: {candidate.profile}

Task:
1. Briefly explain why this candidate is a good fit or not.
2. Highlight specific experience that aligns with the project requirements.
3. Call out any obvious risks or gaps.

Return your answer in 2–3 concise bullet points.
"""
    return prompt.strip()


def fake_llm_explanation(prompt: str) -> str:
    """
    Stub for an LLM call.
    In a real implementation, this would call an LLM API (e.g., OpenAI) with the prompt.
    Here we just return a simple templated explanation to illustrate the flow.
    """
    # In your README, you can show: this is where the LLM call would happen.
    return (
        "- Strong overlap between candidate's experience and project needs based on churn modeling, retention, and experimentation.\n"
        "- Candidate has shipped similar systems end-to-end, suggesting they can move quickly with limited guidance.\n"
        "- Risk: need to validate domain fit and confirm experience with this specific stack and data environment."
    )


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

    ranked = rank_candidates(project, candidates, top_k=3)

    print(f"\nProject: {project.title}\n")
    print("Top matches (retrieval step):\n")
    for idx, (cand, score) in enumerate(ranked, start=1):
        print(f"{idx}. {cand.name:25s}  score={score:.3f}")

    print("\nExample LLM-style explanation for the top candidate (generation step):\n")
    top_candidate, _ = ranked[0]
    prompt = build_llm_prompt(project, top_candidate)
    explanation = fake_llm_explanation(prompt)
    print(explanation)
    print("\n[Note: In a real system, this explanation would be generated by an LLM using the prompt above.]\n")


if __name__ == "__main__":
    main()
