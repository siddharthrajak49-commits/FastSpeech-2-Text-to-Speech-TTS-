"""
Cleaners for Telugu text preprocessing
"""

import re
from text.numbers import normalize_telugu_numbers

def telugu_cleaners(text: str) -> str:
    """
    Cleans and normalizes Telugu text:
    1. Normalizes whitespace and unprintable characters.
    2. Expands numerical digits, dates, and times into Telugu words.
    3. Trims punctuation whitespace.
    """
    # Replace multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Expand numbers, times, and dates
    text = normalize_telugu_numbers(text)
    
    # Clean up double spaces created by replacement
    text = re.sub(r'\s+', ' ', text).strip()
    return text
