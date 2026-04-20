# =============================================================
# target.py — Target Model (Ollama)
# =============================================================
# This file wraps the local Ollama model your agent attacks.
# All other files talk to the target through this class only —
# so if you ever swap Ollama for a different local runner,
# you only change code here.
#
# SETUP:
#   1. Install Ollama: https://ollama.ai
#   2. Pull a model:   ollama pull llama3.2:3b
#   3. Ollama runs automatically when you call this — no need
#      to manually start the server.
# =============================================================

import requests
import json
import time
from typing import Optional


class TargetModel:
    """
    Wraps a locally running Ollama model.
    This is the model your red team agent is attacking.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: the 'target' section from config.yaml
        """
        self.model      = config["model"]
        self.base_url   = config["base_url"].rstrip("/")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens  = config.get("max_tokens", 512)
        self.timeout     = config.get("timeout", 30)

        # Verify Ollama is reachable on init — fail fast with a clear error
        self._check_connection()

    # ----------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------

    def respond(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a prompt to the target model and return its response.

        Args:
            prompt:        The user-turn message (your attack prompt goes here)
            system_prompt: Optional system prompt to set the model's context.
                           Use this to simulate a deployed chatbot with instructions.

        Returns:
            The model's raw text response.

        Raises:
            RuntimeError: if Ollama is unreachable or returns an error.
        """
        messages = []

        # Add system prompt if provided (simulates a real deployment context)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,           # We want the full response at once
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s. "
                f"Try a smaller model or increase timeout in config.yaml"
            )
        except KeyError:
            raise RuntimeError(
                f"Unexpected response format from Ollama: {response.text[:200]}"
            )

    def get_model_info(self) -> dict:
        """
        Returns metadata about the currently loaded model.
        Useful for logging which exact model version was tested.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            models = response.json().get("models", [])

            # Find the model that matches our config
            for m in models:
                if m["name"].startswith(self.model.split(":")[0]):
                    return {
                        "name":     m.get("name"),
                        "size":     m.get("size"),
                        "modified": m.get("modified_at"),
                    }
        except Exception:
            pass

        return {"name": self.model, "size": "unknown", "modified": "unknown"}

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _check_connection(self):
        """Ping Ollama on startup — raises a clear error if it's not running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "\n[TargetModel] Ollama is not running!\n"
                "Start it with:  ollama serve\n"
                f"Expected at:    {self.base_url}"
            )


# =============================================================
# Quick test — run this file directly to verify your setup:
#   python target.py
# =============================================================
if __name__ == "__main__":
    import yaml

    print("Loading config...")
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    print(f"Connecting to Ollama ({config['target']['model']})...")
    target = TargetModel(config["target"])

    info = target.get_model_info()
    print(f"Model info: {info}")

    print("\nSending test prompt...")
    response = target.respond(
        prompt="Say hello in exactly one sentence.",
        system_prompt="You are a helpful assistant."
    )
    print(f"Response: {response}")
    print("\n✓ target.py is working correctly")
