from typing import Tuple, List


def encode(segmented_word: str) -> Tuple[List[str], List[str]]:
    """
    Encodes a segmented word [with '-' betweeen the parts] into a sequence of individual labels.

    Labels used:
    - 'I' (Inside): The character is within a morpheme (no boundary follows).
    - 'S' (Suffix): A single dash '-' follows this character (standard boundary).
    - 'N' (Null): A double dash '--' follows this character (special/null boundary).

    Examples: 
        Input: 'გა-ვ-აკეთ-ებ-თ'
        Output: ['I', 'S', 'S', 'I', 'I', 'I', 'S', 'I', 'S', 'I']

        Input: 'და--წერ--'
        Output: ['I', 'N', 'I', 'I', N]
    """
    labels = []
    i = 0

    while i < len(segmented_word):
        char = segmented_word[i]

        # Guard Clause: Skip boundary markers to isolate the characters
        if char == '-':
            i += 1
            continue

        # Peek at the characters following the current index (exclusive) to determine the label
        next_one = segmented_word[i+1: i+2]
        next_two = segmented_word[i+1: i+3]

        if next_two == '--':
            labels.append('N')
            i += 3  # Move past current char and '--'
        elif next_one == '-':
            labels.append('S')
            i += 2  # Move past current char and '-'
        else:
            labels.append('I')
            i += 1  # Move to next char

    return labels


def decode(chars: List[str], labels: List[str]) -> str:
    """
    Decodes a sequence of individual characters and their labels into a segmented string [with '-' separating the parts].

    Args:
        chars: The raw character list.
        labels: The labels ('I', 'S', or 'N'), eg from the model prediction.

    Example: 
        Input: (
            ['გ', 'ა', 'ვ', 'ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'თ'], 
            ['I', 'S', 'S', 'I', 'I', 'I', 'S', 'I', 'S', 'I']
        )
        Output: 'გა-ვ-აკეთ-ებ-თ'
    """
    if len(chars) != len(labels):
        raise ValueError(
            "Character and Label sequences must be the same length.")

    reconstructed = ""
    for char, label in zip(chars, labels):
        reconstructed += char

        if label == 'S':
            reconstructed += "-"
        elif label == 'N':
            reconstructed += "--"
    return reconstructed
