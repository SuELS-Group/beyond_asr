#!/usr/bin/env python3
from __future__ import annotations
import sys
import asyncio
import inspect
import html
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict 
from googletrans import Translator

LANGUAGE_MAP: Dict[str, str] = {
    "hmong": "hmn",
    "scotts_gaelic": "gd",
    "guarani": "gn",
    "zulu": "zu",
}

def safe_translate(prompt: str, source_lang: str="en", target_lang: str="en") -> str:
    """Safely translate a prompt to a target language using Google Translate API."""
    i = 0
    translated = ""
    while i<3:
        try:
            translator = Translator()
            result = translator.translate(prompt, src=source_lang, dest=target_lang)
            if inspect.isawaitable(result):
                try:
                    result = asyncio.run(result)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(result)
                    finally:
                        loop.close()
            translated = getattr(result, "text", "")
            break
        except Exception as exc:
            print(f"[translate] error for target={target_lang}: {exc}")
            i += 1
            if i >= 3:
                print(f"Translation to {target_lang} failed after 3 attempts.")
    return html.unescape(translated).strip()

def run_translate_attack(prompt: str):
    """Run the translation attack on a given prompt."""
    prompt_translations = []
    for target_lang in LANGUAGE_MAP.items():
        translated = safe_translate(prompt, target_lang=target_lang[1])
        prompt_translations.append({"target_language": target_lang[0],
                                    "translated_text": translated}) 
    return prompt_translations
