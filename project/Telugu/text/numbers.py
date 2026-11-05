"""
Universal Telugu Number and Date Normalization
Soft-coded recursive digit expansion for integers, times, and calendar dates.
"""

import re

DIGITS = {
    0: 'సున్నా',
    1: 'ఒకటి',
    2: 'రెండు',
    3: 'మూడు',
    4: 'నాలుగు',
    5: 'ఐదు',
    6: 'ఆరు',
    7: 'ఏడు',
    8: 'ఎనిమిది',
    9: 'తొమ్మిది'
}

TEENS = {
    10: 'పది',
    11: 'పదకొండు',
    12: 'పన్నెండు',
    13: 'పదమూడు',
    14: 'పద్నాలుగు',
    15: 'పదిహేను',
    16: 'పదహారు',
    17: 'పదిహేడు',
    18: 'పద్దెనిమిది',
    19: 'పందొమ్మిది'
}

TENS = {
    20: 'ఇరవై',
    30: 'ముప్పై',
    40: 'నలభై',
    50: 'యాభై',
    60: 'అరవై',
    70: 'డెబ్బై',
    80: 'ఎనభై',
    90: 'తొంభై'
}

def number_to_telugu(num: int) -> str:
    """
    Recursively converts any integer into its natural Telugu word representation.
    """
    if num < 0:
        return 'మైనస్ ' + number_to_telugu(-num)
    if num in DIGITS:
        return DIGITS[num]
    if num in TEENS:
        return TEENS[num]
    if num in TENS:
        return TENS[num]
    if num < 100:
        ten = (num // 10) * 10
        rem = num % 10
        return f"{TENS[ten]} {DIGITS[rem]}"
    if num < 1000:
        hundreds = num // 100
        rem = num % 100
        h_str = "వంద" if hundreds == 1 else f"{DIGITS[hundreds]} వందల"
        if rem == 0:
            return h_str
        return f"{h_str} {number_to_telugu(rem)}"
    if num < 100000: # Thousands
        thousands = num // 1000
        rem = num % 1000
        t_str = "వెయ్యి" if thousands == 1 else f"{number_to_telugu(thousands)} వేల"
        if rem == 0:
            return t_str
        return f"{t_str} {number_to_telugu(rem)}"
    if num < 10000000: # Lakhs
        lakhs = num // 100000
        rem = num % 100000
        l_str = "లక్ష" if lakhs == 1 else f"{number_to_telugu(lakhs)} లక్షల"
        if rem == 0:
            return l_str
        return f"{l_str} {number_to_telugu(rem)}"
    # Crores
    crores = num // 10000000
    rem = num % 10000000
    c_str = f"{number_to_telugu(crores)} కోట్ల"
    if rem == 0:
        return c_str
    return f"{c_str} {number_to_telugu(rem)}"

def normalize_time(text: str) -> str:
    """
    Normalizes time formats like 10:45 -> పది నలభై ఐదు
    """
    def _time_sub(match):
        h = int(match.group(1))
        m = int(match.group(2))
        h_str = number_to_telugu(h)
        m_str = number_to_telugu(m) if m > 0 else ""
        return f"{h_str} గంటల {m_str}".strip()
    
    return re.sub(r'(\d{1,2}):(\d{2})', _time_sub, text)

def normalize_dates(text: str) -> str:
    """
    Normalizes ordinals like 12వ -> పన్నెండవ (longest match first)
    """
    ordinal_map = {
        '12వ': 'పన్నెండవ', '11వ': 'పదకొండవ', '10వ': 'పదవ',
        '9వ': 'తొమ్మిదవ', '8వ': 'ఎనిమిదవ', '7వ': 'ఏడవ', '6వ': 'ఆరవ',
        '5వ': 'ఐదవ', '4వ': 'నాల్గవ', '3వ': 'మూడవ', '2వ': 'రెండవ', '1వ': 'ఒకటవ'
    }
    for k, v in sorted(ordinal_map.items(), key=lambda x: -len(x[0])):
        text = text.replace(k, v)
    return text

def normalize_telugu_numbers(text: str) -> str:
    """
    Comprehensive pipeline to expand all numbers in Telugu text.
    First expands compound dates/ordinals and times, then standalone digits.
    """
    text = normalize_dates(text)
    text = normalize_time(text)
    
    # Expand any remaining digits (longest match first)
    def _num_sub(match):
        num = int(match.group(0))
        return number_to_telugu(num)
        
    text = re.sub(r'\b\d+\b|\d+', _num_sub, text)
    return text
