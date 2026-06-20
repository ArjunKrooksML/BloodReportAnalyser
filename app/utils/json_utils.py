import json
import re


def parse_llm_json(text: str) -> dict | list:
    """Parse JSON from LLM output, stripping markdown code fences if present."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*$', '', text)
    return json.loads(text.strip())
