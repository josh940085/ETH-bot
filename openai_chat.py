"""Shared OpenAI-compatible chat payload and response helpers."""

import os


def uses_reasoning_chat_model(model_name):
    model = str(model_name or "").strip().lower()
    return model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4"))


def openai_instruction_role(model_name):
    return "developer" if uses_reasoning_chat_model(model_name) else "system"


def build_openai_chat_payload(model_name, messages, temperature=None):
    default_model = (os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini") or "gpt-5-mini").strip()
    model = str(model_name or default_model).strip() or default_model
    payload = {"model": model, "messages": messages}
    if uses_reasoning_chat_model(model):
        reasoning_effort = (os.getenv("OPENAI_REASONING_EFFORT", "low") or "low").strip().lower()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
    elif temperature is not None:
        payload["temperature"] = temperature
    return payload


def extract_openai_chat_text(response_json):
    if not isinstance(response_json, dict):
        return ""
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        ).strip()
    return ""
