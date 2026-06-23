"""
generation/llm_client.py
Unified LLM wrapper — supports Groq, Ollama, Anthropic, OpenAI, AWS Bedrock.

AWS Bedrock uses Bearer token authentication (new Bedrock API key format).
Set AWS_BEARER_TOKEN_BEDROCK and AWS_DEFAULT_REGION in .env.
"""
from __future__ import annotations
import json
import re
import time
import urllib.request
import urllib.error

from loguru import logger
from config.settings import settings


# Global token counter — tracks spend across all calls
_token_usage = {"input": 0, "output": 0}
_TOKEN_LIMIT  = 4_000_000   # 4M input tokens ≈ $0.60 — change as needed

def get_token_usage():
    cost = (_token_usage["input"] / 1_000_000 * 0.15 +
            _token_usage["output"] / 1_000_000 * 0.60)
    return {**_token_usage, "estimated_cost_usd": round(cost, 4)}


class LLMClient:

    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self._client  = self._init_client()
        logger.info(f"LLM provider: {self.provider} | model: {settings.llm_model}")

    # ── Initialise ───────────────────────────────────────────────────────────

    def _init_client(self):
        if self.provider == "bedrock":
            # Bearer token auth — no boto3 needed
            token  = getattr(settings, "aws_bearer_token_bedrock", "")
            region = getattr(settings, "aws_default_region", "us-east-1")
            if not token:
                raise ValueError(
                    "AWS_BEARER_TOKEN_BEDROCK missing in .env\n"
                    "Get it from: AWS Console → Bedrock → API keys"
                )
            return {"token": token, "region": region}

        elif self.provider == "ollama":
            from openai import OpenAI
            base_url = getattr(settings, "ollama_base_url", "http://localhost:11434/v1")
            return OpenAI(base_url=base_url, api_key="ollama")

        elif self.provider == "groq":
            from groq import Groq
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY missing in .env")
            return Groq(api_key=settings.groq_api_key)

        elif self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)

        elif self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=settings.openai_api_key)

        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{self.provider}'\n"
                "Valid: bedrock | ollama | groq | anthropic | openai"
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: list[dict]) -> str:
        try:
            if self.provider == "bedrock":
                return self._bedrock_chat(messages)
            elif self.provider in ("ollama", "openai"):
                return self._openai_compatible_chat(messages)
            elif self.provider == "groq":
                return self._groq_chat_with_retry(messages)
            elif self.provider == "anthropic":
                return self._anthropic_chat(messages)
        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}): {e}")
            raise

    # ── AWS Bedrock — Bearer token ────────────────────────────────────────────

    def _bedrock_chat(self, messages: list[dict]) -> str:
        token  = self._client["token"]
        region = self._client["region"]
        model  = settings.llm_model

        url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke"

        # Separate system message from user messages
        system = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": settings.llm_max_tokens,
            "messages": user_messages,
        }
        if system:
            body["system"] = system

        payload = json.dumps(body).encode()

        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {token}")

        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                result = json.loads(r.read())
                return result["content"][0]["text"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            raise RuntimeError(
                f"Bedrock HTTP {e.code}: {error_body}"
            )

    # ── OpenAI-compatible (Ollama + OpenAI) ──────────────────────────────────

    def _openai_compatible_chat(self, messages: list[dict]) -> str:
        # Check token limit before calling
        if _token_usage["input"] >= _TOKEN_LIMIT:
            cost = get_token_usage()["estimated_cost_usd"]
            raise RuntimeError(
                f"Token limit reached — {_token_usage['input']:,} input tokens used "
                f"(~${cost:.2f}). Increase _TOKEN_LIMIT in llm_client.py to continue."
            )
        response = self._client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        # Track usage
        if hasattr(response, "usage") and response.usage:
            _token_usage["input"]  += response.usage.prompt_tokens
            _token_usage["output"] += response.usage.completion_tokens
            if _token_usage["input"] % 500_000 < 1000:  # log every 500k tokens
                cost = get_token_usage()["estimated_cost_usd"]
                logger.info(f"Token usage: {_token_usage['input']:,} input, ~${cost:.3f} spent")
        return response.choices[0].message.content

    # ── Groq with retry ───────────────────────────────────────────────────────

    def _groq_chat_with_retry(self, messages: list[dict], max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=settings.llm_model,
                    messages=messages,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                err = str(e)
                if "tokens per day" in err or "TPD" in err:
                    raise RuntimeError(
                        f"Groq daily token limit reached.\n"
                        f"Switch to Bedrock: LLM_PROVIDER=bedrock"
                    )
                elif "rate_limit_exceeded" in err or "429" in err:
                    wait_match = re.search(r"try again in ([\d.]+)s", err)
                    wait = float(wait_match.group(1)) if wait_match else 30
                    logger.warning(f"Rate limited — waiting {wait:.0f}s")
                    time.sleep(min(wait, 62) + 1)
                    continue
                else:
                    raise
        raise RuntimeError(f"Groq failed after {max_retries} retries")

    # ── Anthropic direct ─────────────────────────────────────────────────────

    def _anthropic_chat(self, messages: list[dict]) -> str:
        system = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)
        kwargs = dict(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            messages=user_messages,
        )
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text