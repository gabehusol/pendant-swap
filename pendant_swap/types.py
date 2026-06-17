"""Shared dataclasses and named types for pendant-swap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional


class Point(NamedTuple):
    x: int
    y: int


@dataclass
class CheckResult:
    value: float
    target: float
    passed: bool
    label: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.label}: {self.value:.3f} (target {self.target:.3f})"


@dataclass
class QAReport:
    pendant_height_mm: CheckResult
    aspect_ratio: CheckResult
    chain_color: Optional[CheckResult]
    passed: bool
    summary: str
    annotated_image: Any = None  # PIL Image, set when annotate=True

    def __str__(self) -> str:
        return self.summary


@dataclass
class SwapResult:
    final_image: Any  # PIL Image
    chosen_attempt: int
    qa_reports: list[QAReport]
    prompts_used: list[str]
    final_qa: Optional[QAReport] = None   # QA run on final composited image
    gen_image_size: Optional[tuple] = None  # (w, h) of Gemini output


@dataclass
class SwapParams:
    model_path: str
    pendant_path: str
    target_mm: float = 21.0
    ref_px_height: int = 130
    ref_mm: float = 13.0
    hang_x: Optional[int] = None
    hang_y: Optional[int] = None
    rotate_deg: float = 0.0
    top_crop_px: int = 0
    max_retries: int = 4
    mode: str = "generate"   # "composite" | "generate"
    api_key: Optional[str] = None
    tolerance: int = 28
    model_id: str = "gemini-3.1-flash-image"
    extra_prompt: str = ""
    composite_finish: bool = True
