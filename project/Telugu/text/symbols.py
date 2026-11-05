"""
Defines the symbol vocabulary for Telugu FastSpeech 2
"""

from text.telugu_phonemes import TELUGU_PHONEMES

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_silences = ["@sp", "@sil"]

# Export Telugu symbols
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + ["@" + p for p in TELUGU_PHONEMES]
    + _silences
)
