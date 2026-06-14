"""Review module — Phase C/D search query generation and Phase E synthesis."""

from .search_query import generate_queries
from .synthesizer import synthesize_review

__all__ = ["generate_queries", "synthesize_review"]
