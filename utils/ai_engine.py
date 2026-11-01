"""AI engine: ek hi interface se OpenAI, Anthropic aur Gemini use karne ke liye."""

DEFAULT_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "Anthropic": "claude-sonnet-5-5",
    "Gemini": "gemini-2.0-flash",
}


class AIEngineError(Exception):
    """AI call mein koi masla aaye to yeh error raise hota hai."""


class AIEngine:
    def __init__(self, provider="OpenAI", api_key="", model=None):
        if provider not in DEFAULT_MODELS:
            raise AIEngineError(f"Unknown provider: {provider}")
        if not api_key:
            raise AIEngineError("API key nahi di gayi.")
        self.provider = provider
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS[provider]
        self.last_truncated = False

    def generate(self, prompt, system_prompt="You are a helpful assistant.",
                 max_tokens=2000, temperature=0.3):
        self.last_truncated = False
        try:
            if self.provider == "OpenAI":
                return self._openai(prompt, system_prompt, max_tokens, temperature)
            if self.provider == "Anthropic":
                return self._anthropic(prompt, system_prompt, max_tokens, temperature)
            return self._gemini(prompt, system_prompt, max_tokens, temperature)
        except AIEngineError:
            raise
        except Exception as e:
            raise AIEngineError(f"{self.provider} error: {e}") from e

    # alias, agar app.py mein summarize() wagaira use ho
    def summarize(self, text, instructions="Summarize these notes clearly.", **kw):
        return self.generate(f"{instructions}\n\n{text}", **kw)

    def _openai(self, prompt, system_prompt, max_tokens, temperature):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise AIEngineError("'openai' package install nahi hai (requirements.txt check karein).") from e
        client = OpenAI(api_key=self.api_key)
        r = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.last_truncated = r.choices[0].finish_reason == "length"
        return r.choices[0].message.content or ""

    def _anthropic(self, prompt, system_prompt, max_tokens, temperature):
        try:
            import anthropic
        except ImportError as e:
            raise AIEngineError("'anthropic' package install nahi hai (requirements.txt check karein).") from e
        client = anthropic.Anthropic(api_key=self.api_key)
        r = client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.last_truncated = r.stop_reason == "max_tokens"
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

    def _gemini(self, prompt, system_prompt, max_tokens, temperature):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise AIEngineError("'google-generativeai' package install nahi hai (requirements.txt check karein).") from e
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model, system_instruction=system_prompt)
        r = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        try:
            self.last_truncated = "MAX_TOKENS" in str(r.candidates[0].finish_reason)
        except Exception:
            self.last_truncated = False
        return r.text or ""


def was_truncated(engine_or_flag=None):
    """True agar aakhri jawab max_tokens ki wajah se kat gaya ho."""
    if isinstance(engine_or_flag, AIEngine):
        return engine_or_flag.last_truncated
    return bool(engine_or_flag)
