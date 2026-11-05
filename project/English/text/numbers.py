"""
Universal English Number and Time/Date Normalization
Pure Python soft-coded implementation with zero external dependency crashes.
"""

import re

UNITS = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
    11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
    15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen'
}

TENS = {
    20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty',
    60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'
}

def int_to_english(num: int) -> str:
    """
    Recursively converts any integer to English words.
    """
    if num < 0:
        return 'minus ' + int_to_english(-num)
    if num in UNITS:
        return UNITS[num]
    if num in TENS:
        return TENS[num]
    if num < 100:
        ten = (num // 10) * 10
        rem = num % 10
        return f"{TENS[ten]}-{UNITS[rem]}"
    if num < 1000:
        hundreds = num // 100
        rem = num % 100
        h_str = f"{UNITS[hundreds]} hundred"
        if rem == 0:
            return h_str
        return f"{h_str} and {int_to_english(rem)}"
    if num < 1000000:
        thousands = num // 1000
        rem = num % 1000
        # Year expansion e.g. 2026 -> twenty twenty-six
        if 2000 <= num < 2100:
            century = num // 100
            rem_year = num % 100
            if rem_year == 0:
                return "two thousand"
            elif rem_year < 10:
                return f"two thousand {UNITS[rem_year]}"
            else:
                return f"twenty {int_to_english(rem_year)}"
        t_str = f"{int_to_english(thousands)} thousand"
        if rem == 0:
            return t_str
        return f"{t_str} {int_to_english(rem)}"
    return str(num)

def normalize_time(text: str) -> str:
    """
    Normalizes time e.g. 10:45 AM -> ten forty-five a m
    """
    def _time_repl(match):
        h = int(match.group(1))
        m = int(match.group(2))
        ampm = match.group(3).strip() if match.group(3) else ""
        h_words = int_to_english(h)
        if m == 0:
            m_words = "o'clock"
        elif m < 10:
            m_words = f"oh {int_to_english(m)}"
        else:
            m_words = int_to_english(m)
        ampm_words = " a m" if "am" in ampm.lower() else (" p m" if "pm" in ampm.lower() else "")
        return f"{h_words} {m_words}{ampm_words}".strip()

    return re.sub(r'(\b\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b', _time_repl, text)

def normalize_numbers(text: str) -> str:
    """
    Normalizes all numbers, times, and dates in English text.
    """
    text = normalize_time(text)
    
    # Ordinals e.g. 1st, 2nd, 3rd, 12th
    ordinals = {
        '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth',
        '5th': 'fifth', '6th': 'sixth', '7th': 'seventh', '8th': 'eighth',
        '9th': 'ninth', '10th': 'tenth', '11th': 'eleventh', '12th': 'twelfth'
    }
    for k, v in ordinals.items():
        text = re.sub(r'\b' + k + r'\b', v, text, flags=re.IGNORECASE)

    # Remaining numbers
    def _num_repl(match):
        val = int(match.group(0))
        return int_to_english(val)

    text = re.sub(r'\b\d+\b', _num_repl, text)
    return text
