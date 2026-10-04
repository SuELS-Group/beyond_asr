#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict

from deep_translator import GoogleTranslator

LANGUAGE_MAP: Dict[str, str] = {
    "hmong": "hmn",
    "scotts_gaelic": "gd",
    "guarani": "gn",
    "zulu": "zu",
}
    
def safe_translate(prompt: str) -> str:
    """Translate the prompt into a low-resource language."""
    target_lang = random.choice(list(LANGUAGE_MAP.keys()))
    translated_text = GoogleTranslator(source='en', target=LANGUAGE_MAP[target_lang]).translate(prompt)
    return translated_text
