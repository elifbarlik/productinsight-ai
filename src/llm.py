"""Thin Gemini wrapper so the rest of the code (and the tests) depend on a tiny interface:

    generate_text(system, user) -> str
    generate_json(system, user, schema) -> dict     # schema: a pydantic model class
"""
import json
from typing import Type

from pydantic import BaseModel

from src.config import gemini_api_key, llm_model, llm_temperature
from src.embedding_utils import with_retry


class GeminiLLM:
    def __init__(self, model: str = llm_model, temperature: float = llm_temperature):
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai

        self.client = genai.Client(api_key=gemini_api_key)
        self.model = model
        self.temperature = temperature

    def _call(self, system: str, user: str, attempts: int = 5, **config_kwargs) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature, **config_kwargs
        )
        response = with_retry(
            lambda: self.client.models.generate_content(
                model=self.model, contents=user, config=config
            ),
            attempts=attempts,
            base_delay=3.0 if attempts > 2 else 1.0,
        )
        return response.text or ""

    def generate_text(self, system: str, user: str) -> str:
        return self._call(system, user)

    def generate_json(
        self, system: str, user: str, schema: Type[BaseModel], fast: bool = False
    ) -> dict:
        """``fast``: few retries, for optional steps (query parsing) that have a fallback."""
        text = self._call(
            system,
            user,
            attempts=2 if fast else 5,
            response_mime_type="application/json",
            response_schema=schema,
        )
        return json.loads(text)
