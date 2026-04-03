import os
import re

from google import genai


class Generator:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self._model = "gemini-2.5-flash"

    def generate(self, query: str, context: list):
        context_text = "\n".join(context)

        prompt = f"""
        You are a research assistant. Answer using ONLY the context below.
        If the context does not contain enough information, say what is missing briefly.
        Do not cite external knowledge beyond the context.
        Respond in plain paragraph form (no bullet lists or markdown).

        Context:
        {context_text}

        Question:
        {query}

        Answer:
        """

        response = self.client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text

    def validate_answer(self, query: str, context: list, answer: str) -> bool:
        """
        Second LLM pass: YES/NO — is the draft answer fully supported by the retrieved context?
        """
        context_text = "\n".join(context)
        prompt = f"""
        You are a strict grader. Using ONLY the Context below, decide whether the Answer
        is fully supported (no facts in the Answer that are not stated or clearly implied by the Context).

        Reply with exactly one word: YES or NO.

        Context:
        {context_text}

        Question: {query}

        Answer to check:
        {answer}
        """

        response = self.client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        text = (response.text or "").strip().upper()
        return bool(re.match(r"^\s*YES\b", text))

    def summarize(self, text: str, max_input_chars: int = 48_000) -> str:
        """Single-pass summary; long PDFs are truncated for API limits."""
        text = (text or "").strip()
        if not text:
            return "No text to summarize."
        if len(text) > max_input_chars:
            text = text[:max_input_chars] + "\n\n[... truncated for length ...]"

        prompt = f"""
        Summarize the following document for a researcher.
        Use short bullet lines starting with "- ". Cover main goal, methods, and key findings.
        If the text is garbled or empty, say so briefly. Do not invent details.

        Document:
        {text}
        """

        response = self.client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text