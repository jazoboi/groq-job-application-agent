"""
Groq LPU API client with automatic retry and rate limiting.

Provides a clean interface to Groq's ultra-fast inference
endpoint with built-in error handling.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from a Groq inference call."""
    content: str
    model: str
    latency_ms: float
    tokens_used: int


class GroqClient:
    """High-performance client for Groq LPU inference.

    Wraps the Groq SDK with retry logic, latency tracking,
    and structured output parsing.

    Parameters
    ----------
    api_key : str
        Groq API key.
    model : str
        Model identifier (default: llama3-70b-8192).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama3-70b-8192",
    ) -> None:
        self._client = Groq(api_key=api_key)
        self._model = model
        self._total_calls = 0
        self._total_latency_ms = 0.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> InferenceResult:
        """Generate a completion via Groq LPU.

        Parameters
        ----------
        prompt : str
            User prompt.
        system : str
            System prompt for context setting.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum response tokens.

        Returns
        -------
        InferenceResult
            Generated content with latency metrics.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        self._total_calls += 1
        self._total_latency_ms += latency_ms

        return InferenceResult(
            content=response.choices[0].message.content,
            model=self._model,
            latency_ms=round(latency_ms, 2),
            tokens_used=response.usage.total_tokens,
        )

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all calls."""
        if self._total_calls == 0:
            return 0.0
        return self._total_latency_ms / self._total_calls
