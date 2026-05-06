from typing import List, Set


def get_slice(chars: List[str], start: int, end: int) -> str:
    """
    Safely extracts a substring from a list of characters.

    Args:
        chars: The list of characters forming the sequence.
        start: The starting index (inclusive).
        end: The ending index (exclusive).

    Returns:
        A string joined from the specified range, or an empty string if out of bounds.
    """
    if start < 0 or end > len(chars):
        return ""
    return "".join(chars[start:end])


def get_window(chars: List[str], i: int, length: int, direction: str = 'next') -> str:
    """
    Generic helper to grab a sequence of characters starting or ending at index i.

    Args:
        chars: The list of characters.
        i: The current anchor index.
        length: How many characters to include in the window (inclusive of index i).
        direction: 'next' to look forward from i, 'prev' to look backward ending at i.
    """
    if direction == 'next':
        return get_slice(chars, i, i + length)
    return get_slice(chars, i - length + 1, i + 1)


def get_next(chars: List[str], i: int, length: int) -> str:
    """
    Helper to look forward from the current character (inclusive of current index).

    Args:
        chars: The list of characters forming the sequence.
        i: The starting index (inclusive).
        length: The length of the returned string.
    """
    return get_window(chars, i, length, 'next')


def get_prev(chars: List[str], i: int, length: int) -> str:
    """
    Helper to look backward from the current character (inclusive of current index).

    Args:
        chars: The list of characters forming the sequence.
        i: The ending index (inclusive).
        length: The length of the returned string.
    """
    return get_window(chars, i, length, 'prev')


def is_start_of_any_substring(chars: List[str], i: int, targets: Set[str]) -> bool:
    """
    Checks if index i is the starting position of any string within the target set.

    Args:
        chars: The list of characters.
        i: The current index to check.
        targets: A set of strings to search for.
    """
    lengths = {len(t) for t in targets if len(t) > 1}
    return any(get_next(chars, i, l) in targets for l in lengths)


def is_end_of_any_substring(chars: List[str], i: int, targets: Set[str]) -> bool:
    """
    Checks if index i is the ending position of any string within the target set.

    Args:
        chars: The list of characters.
        i: The current index to check.
        targets: A set of strings to search for.
    """
    lengths = {len(t) for t in targets if len(t) > 1}
    return any(get_prev(chars, i, l) in targets for l in lengths)


def is_contained_in_any_substring(chars: List[str], i: int, targets: Set[str]) -> bool:
    """
    Checks if the character at index i is contained within a multi-character 
    string from the target set, but is neither the start nor the end.

    Args:
        chars: The list of characters.
        i: The current index to check.
        targets: A set of strings to search for.
    """
    # Only strings with length > 2 can have an "internal" character
    lengths = {len(t) for t in targets if len(t) > 2}
    for length in lengths:
        # Check all possible internal offsets (1 to length-2)
        for offset in range(1, length - 1):
            if get_slice(chars, i - offset, i - offset + length) in targets:
                return True
    return False


def is_any_substring(item: str, targets: Set[str]) -> bool:
    """
    Checks if a specific item (usually a single character) exists within the target set.
    """
    return item in targets
