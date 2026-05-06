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


def char2features(chars: List[str], i: int, row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a single character into a dictionary of features for the CRF model.

    Args:
        chars: List of all characters in the verb.
        i: The index of the character being featurized.
        row: The source data row containing base_word, screeve, person, word_type, and has_swap
    """
    char = chars[i]
    total_len = len(chars)
    pos_from_end = total_len - i - 1
    pos_ratio = i / (total_len - 1) if total_len > 1 else 0

    features = {
        'bias': 1.0,
        'char': char,
        'screeve': row['screeve'],
        'person': row['person'],
        'word_type': str(row['word_type']),
        'base_word': row['base_word'],
        'swap_flag': row['has_swap'],
        'is_vowel': char in VOWELS,

        # --- Structural Features ---
        'word_len': total_len,
        'is_start': i == 0,
        'is_end': i == total_len - 1,
        'pos_from_end': pos_from_end,
        'pos_ratio': pos_ratio,
        # 'pos_bucket': get_zone(pos_ratio),
        'char_at_end_dist': f"{char}_{pos_from_end}",

        # --- Marker Features ---
        'is_person_char': is_any_substring(char, PERSON_MARKERS),
        'starts_person':  is_start_of_any_substring(chars, i, PERSON_MARKERS),
        'starts_pfsf':    is_start_of_any_substring(chars, i, PFSF_MARKERS),
        'ends_pfsf':      is_end_of_any_substring(chars, i, PFSF_MARKERS),
        'inside_person':  is_contained_in_any_substring(chars, i, PERSON_MARKERS),

        # Surrounding chars (Exclusive of char at index i)
        'prev_char': get_next(chars, i+1, 1),
        'next_char': get_prev(chars, i-1, 1),


        # N-Grams (Inclusive of char at index i)
        'bigram_prev':  get_prev(chars, i, 2),
        'bigram_next':  get_next(chars, i, 2),
        'trigram_prev': get_prev(chars, i, 3),
        'trigram_next': get_next(chars, i, 3),
        'quadgram_prev': get_prev(chars, i, 4),
        'quadgram_next': get_next(chars, i, 4),
    }

    return features
