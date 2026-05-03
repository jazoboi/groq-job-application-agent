"""
Semantic skill matching engine.

Compares extracted job requirements against a candidate's
skill profile using TF-IDF and embedding-based similarity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class SkillMatch:
    """A matched skill with similarity score."""
    requirement: str
    matched_skill: str
    score: float
    category: str = ""


@dataclass
class MatchReport:
    """Full matching report for a job posting."""
    overall_score: float
    matches: list[SkillMatch]
    gaps: list[str]
    strengths: list[str]


class SkillMatcher:
    """Matches candidate skills against job requirements.

    Uses TF-IDF vectorization with cosine similarity for
    fast, explainable skill matching.

    Parameters
    ----------
    match_threshold : float
        Minimum similarity score to consider a match (default: 0.3).
    """

    def __init__(self, match_threshold: float = 0.3) -> None:
        self.threshold = match_threshold
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
        )

    def match(
        self,
        requirements: list[str],
        skills: list[str],
    ) -> MatchReport:
        """Match job requirements against candidate skills.

        Parameters
        ----------
        requirements : list[str]
            Extracted job requirements.
        skills : list[str]
            Candidate's skills and experiences.

        Returns
        -------
        MatchReport
            Detailed matching report with scores and gaps.
        """
        all_texts = requirements + skills
        tfidf_matrix = self._vectorizer.fit_transform(all_texts)

        req_vectors = tfidf_matrix[:len(requirements)]
        skill_vectors = tfidf_matrix[len(requirements):]

        similarity = cosine_similarity(req_vectors, skill_vectors)

        matches = []
        gaps = []

        for i, req in enumerate(requirements):
            best_idx = int(np.argmax(similarity[i]))
            best_score = float(similarity[i, best_idx])

            if best_score >= self.threshold:
                matches.append(SkillMatch(
                    requirement=req,
                    matched_skill=skills[best_idx],
                    score=round(best_score, 3),
                ))
            else:
                gaps.append(req)

        overall = len(matches) / len(requirements) if requirements else 0
        strengths = [m.matched_skill for m in matches if m.score > 0.6]

        return MatchReport(
            overall_score=round(overall * 100, 1),
            matches=sorted(matches, key=lambda m: m.score, reverse=True),
            gaps=gaps,
            strengths=list(set(strengths)),
        )
