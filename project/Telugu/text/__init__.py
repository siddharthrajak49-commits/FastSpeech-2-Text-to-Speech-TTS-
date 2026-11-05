"""
Telugu Text Processing and Phoneme Sequence Conversion
"""

import re
from text.symbols import symbols
from text.cleaners import telugu_cleaners
from text.telugu_phonemes import decompose_telugu_text

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def text_to_sequence(text: str, cleaner_names: list = None) -> list:
    """
    Converts a Telugu text string into a sequence of symbol IDs.
    """
    cleaned_text = telugu_cleaners(text)
    phonemes = decompose_telugu_text(cleaned_text)
    
    sequence = []
    for p in phonemes:
        sym = "@" + p
        if sym in _symbol_to_id:
            sequence.append(_symbol_to_id[sym])
        elif p in _symbol_to_id:
            sequence.append(_symbol_to_id[p])
        elif "@sp" in _symbol_to_id:
            sequence.append(_symbol_to_id["@sp"])
            
    return sequence

def sequence_to_text(sequence: list) -> str:
    """
    Converts a sequence of IDs back to a phoneme string.
    """
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if s.startswith("@"):
                result += " " + s[1:]
            else:
                result += s
    return result.strip()
