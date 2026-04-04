# core python libraries
import os
import json
import re

from groq import Groq
from dotenv import load_dotenv

# load env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_CHARS = 12000


def truncate_text(text: str) -> str:
    if len(text) > MAX_CHARS:
        print(f"[INFO] Text truncated from {len(text)} to {MAX_CHARS} chars")
        return text[:MAX_CHARS] + "\n\n[...document truncated for processing...]"
    return text


def build_prompt(text: str) -> str:
    return f"""You are a document analysis API. Analyze the following document text and return ONLY a valid JSON object with no extra explanation.

The JSON must follow this structure:
{{
  "summary": "",
  "entities": {{
    "names": [],
    "dates": [],
    "organizations": [],
    "amounts": []
  }},
  "sentiment": ""
}}

Document text:
\"\"\"
{text}
\"\"\"
"""


def safe_parse(raw: str) -> dict:
    raw = raw.strip()

    # remove markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise RuntimeError("Failed to parse JSON")


def analyze_document(text: str) -> dict:
    if len(text.strip()) < 30:
        return {
            "summary": "No readable text could be extracted from this document.",
            "entities": {
                "names": [],
                "dates": [],
                "organizations": [],
                "amounts": []
            },
            "sentiment": "Neutral"
        }

    text = text.replace("\x00", "")

    truncated = truncate_text(text)
    prompt = build_prompt(truncated)

    try:
        # retry logic
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                break
            except Exception:
                if attempt == 1:
                    raise
                print("[WARN] Retrying Groq API...")

        raw = response.choices[0].message.content.strip()
        print("[DEBUG TEXT RAW]:", raw[:200])

        result = safe_parse(raw)

        return {
            "summary": result.get("summary", ""),
            "entities": {
                "names": result.get("entities", {}).get("names", []),
                "dates": result.get("entities", {}).get("dates", []),
                "organizations": result.get("entities", {}).get("organizations", []),
                "amounts": result.get("entities", {}).get("amounts", [])
            },
            "sentiment": result.get("sentiment", "Neutral")
        }

    except Exception as e:
        print(f"[Groq API error] {e}")
        raise RuntimeError(f"LLM analysis failed: {str(e)}")


def analyze_image_directly(file_bytes: bytes) -> dict:
    import base64

    image_b64 = base64.b64encode(file_bytes).decode('utf-8')

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """Return ONLY JSON:
{
  "summary": "",
  "entities": {
    "names": [],
    "dates": [],
    "organizations": [],
    "amounts": []
  },
  "sentiment": ""
}"""
                    }
                ]
            }
        ],
        temperature=0.1,
        max_tokens=1000
    )

    raw = response.choices[0].message.content.strip()
    print("[DEBUG VISION RAW]:", raw[:200])

    result = safe_parse(raw)

    return {
        "summary": result.get("summary", ""),
        "entities": {
            "names": result.get("entities", {}).get("names", []),
            "dates": result.get("entities", {}).get("dates", []),
            "organizations": result.get("entities", {}).get("organizations", []),
            "amounts": result.get("entities", {}).get("amounts", [])
        },
        "sentiment": result.get("sentiment", "Neutral")
    }