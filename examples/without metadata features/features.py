from typing import List, Dict, Any
from lib.string_utils import (
    get_next,
    get_prev,
    is_start_of_any_substring,
    is_end_of_any_substring,
    is_contained_in_any_substring,
    is_any_substring
)

# --- CONSTANTS ---
VOWELS = set("აეიოუ")
PERSON_MARKERS = {"ვ", "ს", "მ", "გ", "გვ", "მი", "გი", "უ", "გვი"}
PFSF_MARKERS = {"ებ", "ობ", "ამ", "ავ", "ენ", "ეს"}


def char_to_features(chars: List[str], i: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a single character into a dictionary of features for the CRF model.

    Args:
        chars: List of all characters in the verb.
        i: The index of the character being featurized.
        meta: The metadata, if any (eg base_word, screeve, person, word_type, and has_swap)
    """
    char = chars[i]
    total_len = len(chars)
    pos_from_end = total_len - i - 1
    pos_ratio = i / (total_len - 1) if total_len > 1 else 0

    features = {
        'bias': 1.0,
        'char': char,
        'is_vowel': char in VOWELS,

        'is_start': i == 0,
        'is_end': i == total_len - 1,

        'prev_char': get_next(chars, i+1, 1),
        'next_char': get_prev(chars, i-1, 1),
    }

    return features


def word_to_features(word: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts a clean word (no '-' dashes) and metadata into feature dictionaries.
    """
    chars = list(word)
    return [
        char_to_features(chars, i, meta)
        for i in range(len(chars))
    ]
