#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict

from deep_translator import GoogleTranslator

LANGUAGE_MAP: Dict[str, str] = {
    "zulu": "zu",
    "hmong": "hmn",
    "scots_gaelic": "gd",
    "guarani": "gn",
}
    
def safe_translate(prompt: str) -> str:
    """Translate the prompt into a random low-resource language and back to English."""
    target_lang = random.choice(list(LANGUAGE_MAP.keys()))
    translated_text = GoogleTranslator(source='en', target=LANGUAGE_MAP[target_lang]).translate(prompt)
    return translated_text
