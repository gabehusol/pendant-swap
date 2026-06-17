"""Pluggable AI image generation backend.

Verified against Google AI docs on 2026-06-17:
  https://ai.google.dev/gemini-api/docs/image-generation

Model:   gemini-3.1-flash-image  (Nano Banana 2 — current GA image model)
SDK:     google-genai >= 2.0.0  (2.8.0 is latest stable as of 2026-06-17)
Call:    client.models.generate_content(
             model=MODEL_ID,
             contents=[prompt_str, pil_img, ...],
             config=types.GenerateContentConfig(
                 response_modalities=['TEXT', 'IMAGE']
             )
         )
Extract: for part in response.parts: part.as_image()

The api_key is ALWAYS passed per-call — never read from a module-level global,
never logged, never stored.  The CLI convenience path (env var fallback) is
handled in cli.py, not here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Protocol (interface for all backends)
# ---------------------------------------------------------------------------

class ImageEditor(Protocol):
    def edit(
        self,
        *,
        base_images: "list[PILImage.Image]",
        prompt: str,
        api_key: str,
    ) -> "PILImage.Image":
        """Send base images + prompt to the backend; return the generated image."""
        ...


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

class GeminiEditor:
    """Google Gemini image-editing backend (Nano Banana 2).

    Verified model id and call shape from:
      https://ai.google.dev/gemini-api/docs/image-generation  (2026-06-17)

    The api_key must be passed per call.  It is held only in memory for the
    duration of the request and is never logged or persisted.
    """

    MODEL_ID: str = "gemini-3.1-flash-image"

    # Available image generation models (Nano Banana family)
    MODELS: dict[str, str] = {
        "gemini-3.1-flash-image": "Nano Banana 2 (fast, GA)",
        "gemini-3-pro-image":     "Nano Banana Pro (higher quality)",
        "gemini-2.5-flash-image": "Nano Banana 1 (previous gen)",
    }

    def edit(
        self,
        *,
        base_images: "list[PILImage.Image]",
        prompt: str,
        api_key: str,
        model_id: str = "",
    ) -> "PILImage.Image":
        """Call the Gemini image model and return the generated PIL Image.

        Args:
            base_images: One or more PIL Images sent as context (model photo,
                optional cutout, optional guide).  Order matters: put the model
                photo first, then the cutout, then the guide.
            prompt: Text prompt describing the edit.
            api_key: Gemini API key for this request.  Never hardcoded or logged.

        Returns:
            PIL RGB Image.

        Raises:
            RuntimeError: On auth failure, quota error, safety block, or if no
                image part is returned.  The exception message is user-facing and
                safe (the key is never included in it).
        """
        if not api_key:
            raise RuntimeError(
                "No API key provided.  Pass --api-key or set GEMINI_API_KEY."
            )

        # Import here so the module loads without google-genai installed
        # (composite/QA paths work with no key and no SDK).
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed.  Run: pip install google-genai==2.9.0"
            ) from exc

        active_model = model_id if model_id else self.MODEL_ID
        client = genai.Client(api_key=api_key)

        # Build contents: text prompt followed by all reference images.
        contents: list = [prompt] + list(base_images)

        try:
            response = client.models.generate_content(
                model=active_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
        except Exception as exc:
            raise _translate_error(exc) from exc

        # Extract the first image part from the response.
        # part.as_image() returns google.genai.types.Image (image_bytes, mime_type),
        # not a PIL Image — load the bytes ourselves.
        for part in response.parts:
            if part.inline_data and part.inline_data.data:
                import io as _io
                from PIL import Image as _PIL
                return _PIL.open(_io.BytesIO(part.inline_data.data)).convert("RGB")

        # No image returned — surface any text the model sent as context.
        text_parts = [p.text for p in response.parts if p.text]
        detail = " | ".join(text_parts) if text_parts else "(no detail)"
        raise RuntimeError(
            "Gemini returned no image part.  Model message: %s" % detail
        )


# ---------------------------------------------------------------------------
# Error translation (keep key out of messages)
# ---------------------------------------------------------------------------

def _translate_error(exc: Exception) -> RuntimeError:
    """Convert SDK exceptions to clean, user-facing RuntimeError messages."""
    msg = str(exc)
    low = msg.lower()

    if any(k in low for k in ("api_key", "api key", "401", "unauthorized", "permission")):
        return RuntimeError(
            "Authentication failed — check that your Gemini API key is valid "
            "and has access to the image generation model."
        )
    if any(k in low for k in ("quota", "429", "rate limit", "resource exhausted")):
        return RuntimeError(
            "Gemini quota or rate limit exceeded.  Wait a moment or check your "
            "quota at https://console.cloud.google.com."
        )
    if any(k in low for k in ("safety", "blocked", "harm")):
        return RuntimeError(
            "Request blocked by Gemini safety filters.  "
            "Try rephrasing the prompt or using a less restrictive safety setting."
        )
    # Generic — include message but guarantee the key isn't in it.
    return RuntimeError("Gemini API error: %s" % msg)
