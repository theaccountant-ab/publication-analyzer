"""Schemas for LLM-assisted program discovery.

When no authoritative program is supplied, the tool asks Gemini (grounded in
Google Search) to list the papers presented at a conference. These Pydantic
models are the structured shape it returns. The *publication outcome* of each
paper is determined separately and authoritatively via OpenAlex (see
``openalex.py``), so these models only describe the program itself.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ProgramPaper(BaseModel):
    """One paper on a conference's program."""

    title: str = Field(description="Title of the paper as presented.")
    authors: List[str] = Field(
        default_factory=list,
        description="Author names, if known (used to disambiguate matches).",
    )
    year: Optional[int] = Field(
        default=None,
        description="Calendar year the paper was presented at the conference.",
    )


class ProgramPaperList(BaseModel):
    """Container for the papers found on a conference's program."""

    papers: List[ProgramPaper] = Field(
        default_factory=list,
        description="Every distinct paper presented that could be identified.",
    )
