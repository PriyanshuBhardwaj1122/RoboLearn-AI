# """
# generation/llm_client.py
# Unified LLM wrapper supporting Groq, Anthropic, and OpenAI.
# Includes automatic retry with backoff for Groq rate limits.
# """
# from __future__ import annotations
# import re
# import time
# from loguru import logger
# from config.settings import settings


# class LLMClient:
#     """
#     Unified interface:
#         client.complete(prompt)    → str
#         client.chat(messages)      → str
#     """

#     def __init__(self):
#         self.provider = settings.llm_provider.lower()
#         self._client = self._init_client()
#         logger.info(f"LLM provider: {self.provider} | model: {settings.llm_model}")

#     # ── Initialise ───────────────────────────────────────────────────────────

#     def _init_client(self):
#         if self.provider == "groq":
#             from groq import Groq
#             if not settings.groq_api_key:
#                 raise ValueError(
#                     "GROQ_API_KEY is missing. Add it to your .env file.\n"
#                     "Get a free key at: https://console.groq.com/keys"
#                 )
#             return Groq(api_key=settings.groq_api_key)
#         elif self.provider == "anthropic":
#             import anthropic
#             if not settings.anthropic_api_key:
#                 raise ValueError("ANTHROPIC_API_KEY is missing in .env")
#             return anthropic.Anthropic(api_key=settings.anthropic_api_key)
#         elif self.provider == "openai":
#             from openai import OpenAI
#             if not settings.openai_api_key:
#                 raise ValueError("OPENAI_API_KEY is missing in .env")
#             return OpenAI(api_key=settings.openai_api_key)
#         else:
#             raise ValueError(
#                 f"Unknown LLM_PROVIDER: '{self.provider}'. "
#                 "Valid options: groq | anthropic | openai"
#             )

#     # ── Public API ───────────────────────────────────────────────────────────

#     def complete(self, prompt: str) -> str:
#         return self.chat([{"role": "user", "content": prompt}])

#     def chat(self, messages: list[dict]) -> str:
#         try:
#             if self.provider == "groq":
#                 return self._groq_chat_with_retry(messages)
#             elif self.provider == "anthropic":
#                 return self._anthropic_chat(messages)
#             elif self.provider == "openai":
#                 return self._openai_chat(messages)
#         except Exception as e:
#             logger.error(f"LLM call failed ({self.provider}): {e}")
#             raise

#     # ── Groq with retry ──────────────────────────────────────────────────────

#     def _groq_chat_with_retry(self, messages: list[dict], max_retries: int = 3) -> str:
#         for attempt in range(max_retries):
#             try:
#                 return self._groq_chat(messages)
#             except Exception as e:
#                 err = str(e)
#                 # Daily token limit — cannot retry, raise immediately with clear message
#                 if "tokens per day" in err or "TPD" in err:
#                     wait_match = re.search(r"try again in ([\d.]+)m", err)
#                     wait_mins = float(wait_match.group(1)) if wait_match else "unknown"
#                     raise RuntimeError(
#                         f"⚠️ Groq daily token limit reached.\n"
#                         f"Resets in ~{wait_mins} minutes.\n\n"
#                         f"Quick fix: change LLM_MODEL=llama-3.1-8b-instant in your .env "
#                         f"(500k tokens/day limit instead of 100k)."
#                     )
#                 # Per-minute rate limit — wait and retry
#                 elif "rate_limit_exceeded" in err or "429" in err:
#                     wait_match = re.search(r"try again in ([\d.]+)s", err)
#                     wait_secs = float(wait_match.group(1)) if wait_match else 30
#                     wait_secs = min(wait_secs, 60)  # cap at 60s
#                     logger.warning(f"Rate limited — waiting {wait_secs:.0f}s (attempt {attempt+1}/{max_retries})")
#                     time.sleep(wait_secs + 1)
#                     continue
#                 else:
#                     raise
#         raise RuntimeError(f"Groq call failed after {max_retries} retries")

#     def _groq_chat(self, messages: list[dict]) -> str:
#         response = self._client.chat.completions.create(
#             model=settings.llm_model,
#             messages=messages,
#             temperature=settings.llm_temperature,
#             max_tokens=settings.llm_max_tokens,
#         )
#         return response.choices[0].message.content

#     # ── Anthropic ────────────────────────────────────────────────────────────

#     def _anthropic_chat(self, messages: list[dict]) -> str:
#         system = ""
#         user_messages = []
#         for m in messages:
#             if m["role"] == "system":
#                 system = m["content"]
#             else:
#                 user_messages.append(m)
#         kwargs = dict(
#             model=settings.llm_model,
#             max_tokens=settings.llm_max_tokens,
#             temperature=settings.llm_temperature,
#             messages=user_messages,
#         )
#         if system:
#             kwargs["system"] = system
#         response = self._client.messages.create(**kwargs)
#         return response.content[0].text

#     # ── OpenAI ───────────────────────────────────────────────────────────────

#     def _openai_chat(self, messages: list[dict]) -> str:
#         response = self._client.chat.completions.create(
#             model=settings.llm_model,
#             temperature=settings.llm_temperature,
#             max_tokens=settings.llm_max_tokens,
#             messages=messages,
#         )
#         return response.choices[0].message.content


"""
generation/llm_client.py
Unified LLM wrapper — supports Groq, Ollama (local), Anthropic, OpenAI.

Ollama runs models locally — no API key, no rate limits, no cost.
Install: brew install ollama && ollama serve
Models:  ollama pull qwen2.5:7b   (16GB Mac)
         ollama pull qwen2.5:14b  (24GB Mac)

Set in .env:
    LLM_PROVIDER=ollama
    LLM_MODEL=qwen2.5:7b
    OLLAMA_BASE_URL=http://localhost:11434
"""
from __future__ import annotations
import re
import time
from loguru import logger
from config.settings import settings


class LLMClient:
    """
    Unified interface:
        client.complete(prompt)    → str
        client.chat(messages)      → str
    """

    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self._client  = self._init_client()
        logger.info(f"LLM provider: {self.provider} | model: {settings.llm_model}")

    # ── Initialise ───────────────────────────────────────────────────────────

    def _init_client(self):
        if self.provider == "ollama":
            # Ollama uses OpenAI-compatible API — no key needed
            from openai import OpenAI
            base_url = getattr(settings, "ollama_base_url", "http://localhost:11434/v1")
            return OpenAI(base_url=base_url, api_key="ollama")

        elif self.provider == "groq":
            from groq import Groq
            if not settings.groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY missing in .env\n"
                    "Get free key: https://console.groq.com/keys"
                )
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
                "Valid options: ollama | groq | anthropic | openai"
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: list[dict]) -> str:
        try:
            if self.provider in ("ollama", "openai"):
                return self._openai_compatible_chat(messages)
            elif self.provider == "groq":
                return self._groq_chat_with_retry(messages)
            elif self.provider == "anthropic":
                return self._anthropic_chat(messages)
        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}): {e}")
            raise

    # ── Ollama + OpenAI (same API format) ────────────────────────────────────

    def _openai_compatible_chat(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return response.choices[0].message.content

    # ── Groq with smart retry ─────────────────────────────────────────────────

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
                # Daily token limit — cannot retry
                if "tokens per day" in err or "TPD" in err:
                    wait_match = re.search(r"try again in ([\d.]+)m", err)
                    wait_mins  = float(wait_match.group(1)) if wait_match else "unknown"
                    raise RuntimeError(
                        f"⚠️ Groq daily token limit reached.\n"
                        f"Resets in ~{wait_mins} minutes.\n\n"
                        f"Switch to Ollama:\n"
                        f"  LLM_PROVIDER=ollama\n"
                        f"  LLM_MODEL=qwen2.5:7b"
                    )
                # Per-minute rate limit — wait and retry
                elif "rate_limit_exceeded" in err or "429" in err:
                    wait_match = re.search(r"try again in ([\d.]+)s", err)
                    wait_secs  = float(wait_match.group(1)) if wait_match else 30
                    wait_secs  = min(wait_secs, 62)
                    logger.warning(f"Rate limited — waiting {wait_secs:.0f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_secs + 1)
                    continue
                else:
                    raise
        raise RuntimeError(f"Groq call failed after {max_retries} retries")

    # ── Anthropic ────────────────────────────────────────────────────────────

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
            temperature=settings.llm_temperature,
            messages=user_messages,
        )
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text