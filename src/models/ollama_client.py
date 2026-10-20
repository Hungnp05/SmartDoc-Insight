"""
Ollama Client
─────────────
Thin wrapper around the Ollama REST API.
Handles: text generation, vision queries, embeddings, streaming.
"""

import json
import logging
import requests
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


def _normalize_model_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


class OllamaClient:
    """
    Unified client for all Ollama model interactions.
    """

    def __init__(self, config):
        self.cfg = config.ollama
        self.base_url = self.cfg.base_url.rstrip("/")
        self._legacy_embed_api = False
        self._check_connection()

    # Connection

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            logger.info(f"Ollama connected. Available models: {models}")
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}. Make sure Ollama is running.")

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def is_model_available(self, model_name: str, available: Optional[list] = None) -> bool:
        if available is None:
            available = self.list_models()
        wanted = _normalize_model_name(model_name)
        return any(_normalize_model_name(m) == wanted for m in available)

    # Text Generation

    def query(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single-turn text completion."""
        payload = self._build_payload(
            model=model or self.cfg.llm_model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.cfg.timeout,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama timeout after {self.cfg.timeout}s")
        except Exception as e:
            raise OllamaError(f"Ollama query failed: {e}")

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Streaming text generation — yields tokens."""
        payload = self._build_payload(
            model=model or self.cfg.llm_model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.cfg.timeout,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise OllamaError(data["error"])
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama timeout after {self.cfg.timeout}s")
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Ollama stream failed: {e}")

    # Vision (LLaVA)

    def vision_query(
        self,
        prompt: str,
        image_base64: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Send image + text prompt to LLaVA vision model.
        image_base64: base64-encoded image string
        """
        payload = {
            "model": model or self.cfg.vision_model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            },
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.cfg.timeout * 2,  # Vision takes longer
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            raise OllamaError(f"Vision query failed: {e}")

    # Embeddings

    def embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        """
        Generate embeddings for a list of texts (one batched request).
        Raises OllamaError instead of returning placeholder vectors, so a failed
        embedding never ends up in the vector store.
        """
        if not texts:
            return []
        model = model or self.cfg.embed_model

        if not self._legacy_embed_api:
            r = self._post_embed("/api/embed", {"model": model, "input": texts})
            if r.status_code != 404:
                self._raise_for_status(r, "Embedding failed")
                embeddings = r.json().get("embeddings", [])
                if len(embeddings) != len(texts):
                    raise OllamaError(
                        f"Embedding failed: expected {len(texts)} vectors, got {len(embeddings)}"
                    )
                return embeddings
            # Ollama < 0.3 has no batch endpoint.
            self._legacy_embed_api = True

        embeddings = []
        for text in texts:
            r = self._post_embed("/api/embeddings", {"model": model, "prompt": text})
            self._raise_for_status(r, "Embedding failed")
            emb = r.json().get("embedding")
            if not emb:
                raise OllamaError("Embedding failed: empty vector returned")
            embeddings.append(emb)
        return embeddings

    def _post_embed(self, path: str, payload: dict):
        try:
            return requests.post(f"{self.base_url}{path}", json=payload, timeout=self.cfg.timeout)
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Embedding failed: {e}")

    @staticmethod
    def _raise_for_status(r, message: str):
        if r.ok:
            return
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        raise OllamaError(f"{message}: HTTP {r.status_code} {detail}")

    # Helpers

    def _build_payload(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_predict": self.cfg.max_tokens if max_tokens is None else max_tokens,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if system:
            payload["system"] = system
        return payload

    def pull_model(self, model_name: str, progress_callback=None) -> bool:
        """Pull a model from Ollama registry."""
        try:
            with requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line and progress_callback:
                        data = json.loads(line)
                        status = data.get("status", "")
                        progress_callback(status)
                        if status == "success":
                            return True
            return True
        except Exception as e:
            logger.error(f"Model pull failed: {e}")
            return False
