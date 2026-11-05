"""
Universal Telugu Phonetic and Akshara Decomposition Engine
Soft-coded phonological decomposition based on Unicode standard (U+0C00 - U+0C7F)
"""

import re
import unicodedata

# 1. Independent Vowels (అచ్చులు)
INDEPENDENT_VOWELS = {
    '\u0C05': 'a',
    '\u0C06': 'aa',
    '\u0C07': 'i',
    '\u0C08': 'ii',
    '\u0C09': 'u',
    '\u0C0A': 'uu',
    '\u0C0B': 'ru',
    '\u0C60': 'ruu',
    '\u0C0C': 'lu',
    '\u0C61': 'luu',
    '\u0C0E': 'e',
    '\u0C0F': 'ee',
    '\u0C10': 'ai',
    '\u0C12': 'o',
    '\u0C13': 'oo',
    '\u0C14': 'au'
}

# 2. Consonants (హల్లులు) - inherent 'a'
CONSONANTS = {
    '\u0C15': 'k',   '\u0C16': 'kh',  '\u0C17': 'g',   '\u0C18': 'gh',  '\u0C19': 'ng',
    '\u0C1A': 'ch',  '\u0C1B': 'chh', '\u0C1C': 'j',   '\u0C1D': 'jh',  '\u0C1E': 'ny',
    '\u0C1F': 'tt',  '\u0C20': 'tth', '\u0C21': 'dd',  '\u0C22': 'ddh', '\u0C23': 'nn',
    '\u0C24': 't',   '\u0C25': 'th',  '\u0C26': 'd',   '\u0C27': 'dh',  '\u0C28': 'n',
    '\u0C2A': 'p',   '\u0C2B': 'ph',  '\u0C2C': 'b',   '\u0C2D': 'bh',  '\u0C2E': 'm',
    '\u0C2F': 'y',   '\u0C30': 'r',   '\u0C31': 'rr',  '\u0C32': 'l',   '\u0C33': 'll',
    '\u0C34': 'zh',  '\u0C35': 'v',   '\u0C36': 'sh',  '\u0C37': 'ss',  '\u0C38': 's',
    '\u0C39': 'h'
}

# 3. Dependent Vowel Signs / Matras (గుణింతాలు)
MATRAS = {
    '\u0C3E': 'aa',
    '\u0C3F': 'i',
    '\u0C40': 'ii',
    '\u0C41': 'u',
    '\u0C42': 'uu',
    '\u0C43': 'ru',
    '\u0C44': 'ruu',
    '\u0C46': 'e',
    '\u0C47': 'ee',
    '\u0C48': 'ai',
    '\u0C4A': 'o',
    '\u0C4B': 'oo',
    '\u0C4C': 'au'
}

VIRAMA = '\u0C4D'      # ్ (Halant / Pollu)
ANUSVARA = '\u0C02'    # ం (Sunna)
VISARGA = '\u0C03'     # ః
ARDHANUSVARA = '\u0C01'# ఁ

# All valid phoneme tokens
TELUGU_PHONEMES = sorted(list(set(
    list(INDEPENDENT_VOWELS.values()) +
    list(CONSONANTS.values()) +
    list(MATRAS.values()) +
    ['m_nasal', 'h_asp', 'sil', 'sp']
)))

def decompose_telugu_text(text: str) -> list:
    """
    Universally decomposes any Telugu text into phonetic tokens.
    Correctly resolves:
      - Inherent vowel suppression by virama (్)
      - Complex conjuncts (e.g. న్య = n + y + a, ర్ఫ = r + ph + a, శ్లే = sh + l + ee)
      - Geminates (e.g. పట్టు = p + a + tt + tt + u)
      - Anusvara context (ం)
    """
    text = unicodedata.normalize('NFC', text)
    tokens = []
    i = 0
    n = len(text)
    
    while i < n:
        char = text[i]
        
        # 1. Independent Vowel
        if char in INDEPENDENT_VOWELS:
            tokens.append(INDEPENDENT_VOWELS[char])
            i += 1
            continue
            
        # 2. Consonant
        if char in CONSONANTS:
            cons_phone = CONSONANTS[char]
            # Check next character
            if i + 1 < n:
                next_char = text[i + 1]
                if next_char == VIRAMA:
                    # Consonant without inherent vowel (part of conjunct or word-final)
                    tokens.append(cons_phone)
                    i += 2
                    continue
                elif next_char in MATRAS:
                    # Consonant + Matra
                    tokens.append(cons_phone)
                    tokens.append(MATRAS[next_char])
                    i += 2
                    continue
                elif next_char == ANUSVARA:
                    # Consonant + Inherent 'a' + Anusvara
                    tokens.append(cons_phone)
                    tokens.append('a')
                    tokens.append('m_nasal')
                    i += 2
                    continue
                elif next_char == VISARGA:
                    tokens.append(cons_phone)
                    tokens.append('a')
                    tokens.append('h_asp')
                    i += 2
                    continue
            
            # Default: Consonant with inherent 'a'
            tokens.append(cons_phone)
            tokens.append('a')
            i += 1
            continue
            
        # 3. Anusvara standalone / following vowel
        if char == ANUSVARA:
            tokens.append('m_nasal')
            i += 1
            continue
            
        # 4. Visarga
        if char == VISARGA:
            tokens.append('h_asp')
            i += 1
            continue
            
        # 5. Spaces & Punctuation
        if char == ' ':
            tokens.append('sp')
            i += 1
            continue
        elif char in ',.;:!?':
            tokens.append('sil')
            i += 1
            continue
            
        # Skip unknown or format chars
        i += 1
        
    return tokens
