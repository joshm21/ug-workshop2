from typing import Dict, Tuple, List, Any


def encode_sequence(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Encodes a segmented string [with '-' betweeen the parts] into a sequence of individual characters and their labels.

    Labels used:
    - 'I' (Inside): The character is within a morpheme (no boundary follows).
    - 'S' (Suffix): A single dash '-' follows this character (standard boundary).
    - 'N' (Null): A double dash '--' follows this character (special/null boundary).

    Example: 
        Input: 'გა-ვ-აკეთ-ებ-თ'
        Output: (
            ['გ', 'ა', 'ვ', 'ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'თ'], 
            ['I', 'S', 'S', 'I', 'I', 'I', 'S', 'I', 'S', 'I']
        )
    """
    split_form = str(row['split_form'])
    chars, labels = [], []
    i = 0

    while i < len(split_form):
        char = split_form[i]

        # Guard Clause: Skip boundary markers to isolate the characters
        if char == '-':
            i += 1
            continue

        chars.append(char)

        # Peek at the characters following the current index (exclusive) to determine the label
        next_one = split_form[i+1: i+2]
        next_two = split_form[i+1: i+3]

        if next_two == '--':
            labels.append('N')
            i += 3  # Move past current char and '--'
        elif next_one == '-':
            labels.append('S')
            i += 2  # Move past current char and '-'
        else:
            labels.append('I')
            i += 1  # Move to next char

    return chars, labels


def decode_sequence(chars: List[str], labels: List[str]) -> str:
    """
    Decodes a sequence of individual characters and their labels into a segmented string [with '-' separating the parts].

    Args:
        chars: The raw character list.
        labels: The labels ('I', 'S', or 'N'), eg from the model prediction.

    Example: 
        Input: 'გა-ვ-აკეთ-ებ-თ'
        Output: (
            ['გ', 'ა', 'ვ', 'ა', 'კ', 'ე', 'თ', 'ე', 'ბ', 'თ'], 
            ['I', 'S', 'S', 'I', 'I', 'I', 'S', 'I', 'S', 'I']
        )
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
