"""
llm_client.py — Ollama-based LLM for legal question answering.

Uses the local Ollama API for fully offline, privacy-preserving inference.
Supports streaming responses via WebSocket.
"""
import json
import sys
import httpx
from pathlib import Path
from typing import AsyncGenerator, Generator
from rich.console import Console

console = Console()

LEGAL_SYSTEM_PROMPT = """Tu es un assistant juridique expert spécialisé dans le droit de l'Union européenne.
Tu analyses des traités, des articles juridiques et des actes législatifs européens.

Tes règles:
1. Base UNIQUEMENT tes réponses sur le contexte juridique fourni
2. Cite toujours les articles et traités sources (ex: "Article 6 TUE", "Article 123 TFUE")
3. Si une information n'est pas dans le contexte, dis-le clairement
4. Utilise un langage juridique précis mais accessible
5. Structure tes réponses de manière claire: explication + sources
6. En cas d'ambiguïté, mentionne les différentes interprétations possibles

Format de réponse:
- Commence par une réponse directe à la question
- Développe avec les détails juridiques pertinents
- Termine par les sources citées (liste d'articles)
"""

QUERY_TEMPLATE = """## Contexte juridique

{context}

---

## Question

{question}

## Réponse

"""


class OllamaClient:
    """Client for local Ollama LLM API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def generate(
        self,
        prompt: str,
        system: str = LEGAL_SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response (blocking)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except httpx.ConnectError:
            return self._ollama_not_running_message()
        except Exception as e:
            return f"Erreur LLM: {e}"

    def generate_stream(
        self,
        prompt: str,
        system: str = LEGAL_SYSTEM_PROMPT,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """Generate a streaming response (sync generator)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "options": {"temperature": temperature},
        }

        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            ) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            yield self._ollama_not_running_message()

    async def agenerate(
        self,
        prompt: str,
        system: str = LEGAL_SYSTEM_PROMPT,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Async generate a response."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
            except httpx.ConnectError:
                return self._ollama_not_running_message()
            except Exception as e:
                return f"Erreur LLM: {e}"

    async def agenerate_stream(
        self,
        prompt: str,
        system: str = LEGAL_SYSTEM_PROMPT,
        temperature: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        """Async streaming generator."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                token = data.get("message", {}).get("content", "")
                                if token:
                                    yield token
                                if data.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
            except httpx.ConnectError:
                yield self._ollama_not_running_message()

    def _ollama_not_running_message(self) -> str:
        return (
            "⚠️ Ollama n'est pas disponible. "
            "Veuillez démarrer Ollama: `ollama serve` "
            f"puis télécharger le modèle: `ollama pull {self.model}`"
        )


class PromptBuilder:
    """Builds prompts for legal QA from retrieved context."""

    @staticmethod
    def build_qa_prompt(question: str, context: str) -> str:
        """Build the main QA prompt."""
        return QUERY_TEMPLATE.format(context=context, question=question)

    @staticmethod
    def build_conversation_prompt(
        question: str,
        context: str,
        history: list[dict],
    ) -> str:
        """Build prompt with conversation history."""
        history_str = ""
        for turn in history[-4:]:  # Keep last 4 turns
            role = "Utilisateur" if turn["role"] == "user" else "Assistant"
            history_str += f"\n{role}: {turn['content']}\n"

        return f"""## Contexte juridique

{context}

---

## Historique de conversation
{history_str}
## Question actuelle

{question}

## Réponse

"""


# ─── Singleton ────────────────────────────────────────────────────────────────

_llm: OllamaClient | None = None


def get_llm() -> OllamaClient:
    global _llm
    if _llm is None:
        from backend.config import settings
        _llm = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    return _llm


if __name__ == "__main__":
    llm = OllamaClient()
    if llm.is_available():
        models = llm.list_models()
        print(f"Available models: {models}")

        response = llm.generate(
            prompt="Qu'est-ce que le Traité de Lisbonne ?",
            system="Tu es un expert en droit européen. Réponds brièvement.",
        )
        print(f"\nResponse:\n{response}")
    else:
        print("Ollama not running. Start with: ollama serve")
