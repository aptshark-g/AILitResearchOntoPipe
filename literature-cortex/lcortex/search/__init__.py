"""Search adapters (Phase A) — multi-source academic paper search.

Modules
───────
- arxiv.py:       arXiv API search (streaming + batch)
- openalex.py:    OpenAlex API search
- multi_level.py: 3-level search filter (API → keyword → LLM)
- dedup.py:       DOI→arXiv→title deduplication
- stream.py:      JSONL streaming output
- adapter.py:     Common search adapter interface
"""

from .arxiv import search_arxiv, search_arxiv_stream
from .multi_level import (
    multi_level_search,
    search_level_1,
    filter_level_2,
    filter_level_3,
)
from .stream import stream_to_jsonl, read_jsonl_stream

__all__ = [
    "search_arxiv",
    "search_arxiv_stream",
    "multi_level_search",
    "search_level_1",
    "filter_level_2",
    "filter_level_3",
    "stream_to_jsonl",
    "read_jsonl_stream",
]
