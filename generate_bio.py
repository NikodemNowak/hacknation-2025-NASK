import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Iterator

from tqdm import tqdm


def read_file(filepath: str) -> str:
    """
    Reads UTF-8 text file and return its content.

    Args:
        filepath: Path to the file

    Returns:
        File content as string

    Raises:
        RuntimeError: If file cannot be read (wraps original exception)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        raise RuntimeError(f"Could not read file {filepath}") from e


def tokenize(text: str) -> list[str]:
    """
    Splits text into tokens.

    Args:
        text: Input text string

    Returns:
        List of tokens
    """
    return re.findall(r'\S+', text)


def find_placeholder_positions(anon_text: str) -> list[dict]:
    """
    Finds all placeholders in anonymized text.

    Args:
        anon_text: Anonymized text string

    Returns:
        List of dictionaries with placeholder info (start, end, type, full)
    """
    pattern = r'\[([a-z\-]+)\]'
    placeholders = []
    for match in re.finditer(pattern, anon_text):
        placeholders.append(
            {
                'start': match.start(),
                'end': match.end(),
                'type': match.group(1),
                'full': match.group(0),
            }
        )
    return placeholders


def is_valid_extracted_value(
    value: str, placeholder_type: str, has_close_placeholder: bool
) -> bool:
    """
    Validates extracted value to avoid common errors.

    Args:
        value: Extracted original value
        placeholder_type: Type of the placeholder (e.g., 'name', 'email')
        has_close_placeholder: Whether there's another placeholder nearby in the anonymized text

    Returns:
        True if the value is valid, False otherwise
    """
    if '(' in value and ')' not in value:
        return False
    if ')' in value and '(' not in value:
        return False

    if (
        value.endswith('(')
        or value.endswith('[')
        or value.startswith(')')
        or value.startswith(']')
    ):
        return False

    if placeholder_type in ['email', 'pesel']:
        if ' ' in value:
            return False

        if '(' in value or ')' in value:
            return False

    if placeholder_type == 'phone':
        digits_and_spaces = sum(1 for c in value if c.isdigit() or c.isspace())
        if digits_and_spaces < len(value) * 0.66:
            return False

    if placeholder_type in ['age', 'number']:
        if not any(c.isdigit() for c in value):
            return False

    if has_close_placeholder:
        if placeholder_type in [
            'name',
            'surname',
            'relative',
            'street',
            'city',
            'company',
        ]:
            if value.count(' ') > 3:
                return False
        if ';' in value:
            return False

    return True


def process_single_placeholder(
    placeholder: dict, anon_text: str, original_text: str
) -> list[dict]:
    """
    Process a single placeholder to find its original value.

    Args:
        placeholder: Dictionary with placeholder info (start, end, type, full)
        anon_text: Anonymized text string
        original_text: Original text string

    Returns:
        List of pairs (original, anonymized) found for this placeholder
    """
    pairs = []

    try:
        before = re.escape(
            anon_text[max(0, placeholder['start'] - 30) : placeholder['start']]
        )

        next_placeholder_start = len(anon_text)
        next_bracket = anon_text.find('[', placeholder['end'])
        if next_bracket != -1:
            next_placeholder_start = next_bracket

        after_end = min(
            len(anon_text), placeholder['end'] + 30, next_placeholder_start
        )
        after_text = anon_text[placeholder['end'] : after_end]

        has_close_placeholder = (
            next_placeholder_start < placeholder['end'] + 15
        )

        if has_close_placeholder and after_text.strip():
            after = re.escape(after_text)
        else:
            after = re.escape(after_text)

        if before and after:  # AI GENERATED
            boundary_chars = set()
            for char in after_text:
                if char in '()[]{};<>,.:!?':
                    boundary_chars.add(char)

            if boundary_chars and has_close_placeholder:
                escaped_chars = ''.join(re.escape(c) for c in boundary_chars)
                capture_pattern = f'([^{escaped_chars}]+?)'
            elif has_close_placeholder:
                capture_pattern = r'(\S+(?:\s+\S+){0,2}?)'
            else:
                capture_pattern = r'([^\[\]]+?)'

            pattern = before + capture_pattern + after
            pattern = pattern.replace(r'\ ', r'\s+')

            matches = re.finditer(pattern, original_text, re.DOTALL)
            for match in matches:
                original_value = match.group(1).strip()
                original_value = re.sub(r'\s+', ' ', original_value)
                original_value = original_value.rstrip('.,;: ')

                if len(original_value) > 100:
                    continue

                if '\n' in original_value:
                    continue

                if not original_value:
                    continue

                if not is_valid_extracted_value(
                    original_value, placeholder['type'], has_close_placeholder
                ):
                    continue

                if has_close_placeholder:
                    if original_value.count(' ') > 4:
                        continue

                pairs.append(
                    {
                        'original': original_value,
                        'anonymized': placeholder['type'],
                    }
                )
                break
    except Exception:
        pass

    return pairs


def extract_pairs(
    original_text: str,
    anon_text: str,
    max_workers: int = 8,
    verbose: bool = False,
) -> list[dict]:
    """
    Extract original-anonymized pairs by comparing texts using context matching.

    Args:
        original_text: Original text string
        anon_text: Anonymized text string
        max_workers: Number of worker threads for parallel processing
        verbose: Show progress bar

    Returns:
        List of pairs (original, anonymized, method) found
    """
    placeholders = find_placeholder_positions(anon_text)
    all_pairs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_placeholder = {
            executor.submit(
                process_single_placeholder, ph, anon_text, original_text
            ): ph
            for ph in placeholders
        }

        iterator: Iterator = as_completed(future_to_placeholder)
        if verbose:
            iterator = tqdm(
                iterator,
                total=len(placeholders),
                desc="Context matching",
                unit="placeholder",
            )

        for future in iterator:
            try:
                pairs = future.result()
                all_pairs.extend(pairs)
            except Exception:
                pass

    for pair in all_pairs:
        pair['method'] = 'context_matching'

    return all_pairs


def process_opcode_chunk(
    opcode_data: tuple,
    orig_tokens: list[str],
    anon_tokens: list[str],
    placeholder_pattern: re.Pattern,
) -> list[dict]:
    """
    Process a single opcode (replace operation) to extract pairs.

    Args:
        opcode_data: Tuple from SequenceMatcher (tag, i1, i2, j1, j2)
        orig_tokens: Original text tokens
        anon_tokens: Anonymized text tokens
        placeholder_pattern: Compiled regex pattern for placeholders

    Returns:
        List of pairs found in this opcode chunk
    """
    pairs = []
    tag, i1, i2, j1, j2 = opcode_data

    try:
        if tag == 'replace':
            anon_part = ' '.join(anon_tokens[j1:j2])
            matches = list(placeholder_pattern.finditer(anon_part))

            if len(matches) == 1:  # AI GENERATED
                match = matches[0]
                placeholder_type = match.group(1)

                orig_part = ' '.join(orig_tokens[i1:i2])
                orig_part = re.sub(r'\s+', ' ', orig_part).strip()
                orig_part = re.sub(r'[,.\s]+$', '', orig_part)
                orig_part = re.sub(r'^[,.\s]+', '', orig_part)

                if orig_part and len(orig_part) < 100:
                    if is_valid_extracted_value(
                        orig_part, placeholder_type, False
                    ):
                        pairs.append(
                            {
                                'original': orig_part,
                                'anonymized': placeholder_type,
                            }
                        )
    except Exception:
        pass

    return pairs


def align_and_extract(
    original_text: str,
    anon_text: str,
    max_workers: int = 8,
    verbose: bool = False,
) -> list[dict]:
    """
    Extract pairs using word-by-word token alignment.

    Args:
        original_text: Original text string
        anon_text: Anonymized text string
        max_workers: Number of worker threads for parallel processing
        verbose: Show progress bar

    Returns:
        List of pairs (original, anonymized, method) found
    """
    orig_tokens = tokenize(original_text)
    anon_tokens = tokenize(anon_text)

    placeholder_pattern = re.compile(r'\[([a-z\-]+)\]')

    matcher = SequenceMatcher(None, orig_tokens, anon_tokens)
    opcodes = list(matcher.get_opcodes())

    all_pairs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_opcode = {
            executor.submit(
                process_opcode_chunk,
                opcode,
                orig_tokens,
                anon_tokens,
                placeholder_pattern,
            ): opcode
            for opcode in opcodes
        }

        iterator: Iterator = as_completed(future_to_opcode)
        if verbose:
            iterator = tqdm(
                iterator,
                total=len(opcodes),
                desc="Token alignment",
                unit="opcode",
            )

        for future in iterator:
            try:
                pairs = future.result()
                all_pairs.extend(pairs)
            except Exception:
                pass

    for pair in all_pairs:
        pair['method'] = 'token_alignment'

    return all_pairs


def deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    """
    Remove duplicate pairs and keep most common ones.

    Prioritizes token alignment results over context matching.

    Args:
        pairs: List of pairs with 'original', 'anonymized', and 'method' keys

    Returns:
        List of deduplicated pairs with count and method info
    """
    unique_pairs = {}
    pair_sources = {}

    for pair in pairs:
        key = (pair['original'], pair['anonymized'])
        weight = 2 if pair.get('method') == 'token_alignment' else 1
        unique_pairs[key] = unique_pairs.get(key, 0) + weight

        if key not in pair_sources or weight > 1:
            pair_sources[key] = pair.get('method', 'context')

    result = []
    for (orig, anon), count in sorted(
        unique_pairs.items(), key=lambda x: -x[1]
    ):
        result.append(
            {
                'original': orig,
                'anonymized': anon,
                'count': count,
                'method': pair_sources.get((orig, anon), 'unknown'),
            }
        )

    return result


def tokenize_with_positions(text: str, verbose: bool = False) -> list[dict]:
    """
    Tokenizes text and returns tokens with their character positions.

    Args:
        text: Input text string

    Returns:
        List of dictionaries with 'text', 'start', and 'end' keys
    """
    tokens = []
    matches = list(re.finditer(r'\S+', text))

    iterators = matches
    if verbose:
        iterators = tqdm(matches, desc="Tokenizing text", unit="token")

    for match in iterators:
        tokens.append(
            {
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
            }
        )
    return tokens


def find_entity_spans(
    original_text: str, pairs: list[dict], verbose: bool = False
) -> list[dict]:
    """
    Finds spans of entities in the original text based on extracted pairs.

    Args:
        original_text: Original text string
        pairs: List of pairs with 'original' and 'anonymized' keys
        verbose: Show progress bar

    Returns:
        List of entities with 'start', 'end', 'category', and 'text' keys
    """
    entities = []
    occupied_ranges = []

    sorted_pairs = sorted(pairs, key=lambda x: -len(x['original']))

    iterator = sorted_pairs
    if verbose:
        iterator = tqdm(sorted_pairs, desc="Finding entity spans", unit="pair")

    for pair in iterator:
        pattern = re.escape(pair['original'])
        for match in re.finditer(pattern, original_text):
            start, end = match.start(), match.end()

            overlap = False
            for occ_start, occ_end in occupied_ranges:
                if not (end <= occ_start or start >= occ_end):
                    overlap = True
                    break

            if not overlap:
                entities.append(
                    {
                        'start': start,
                        'end': end,
                        'category': pair['anonymized'],
                        'text': pair['original'],
                    }
                )
                occupied_ranges.append((start, end))

    return sorted(entities, key=lambda x: x['start'])


def assign_bio_tags_chunk(
    tokens: list[dict], entities_chunk: list[dict]
) -> dict:
    """
    Assigns BIO tags to tokens for a chunk of entities.

    Args:
        tokens: List of tokens with 'text', 'start', and 'end' keys
        entities_chunk: List of entities with 'start', 'end', and 'category' keys

    Returns:
        Dictionary mapping token indices to BIO tags
    """
    tags_updates = {}

    for entity in entities_chunk:
        entity_start = entity['start']
        entity_end = entity['end']
        category = entity['category']

        first_token_in_entity = True
        for i, token in enumerate(tokens):
            token_start = token['start']
            token_end = token['end']

            if token_end <= entity_start:
                continue
            if token_start >= entity_end:
                break

            if token_start >= entity_start and token_end <= entity_end:
                if first_token_in_entity:
                    tags_updates[i] = f'B-{category}'
                    first_token_in_entity = False
                else:
                    tags_updates[i] = f'I-{category}'
            elif token_start < entity_end and token_end > entity_start:
                if first_token_in_entity:
                    tags_updates[i] = f'B-{category}'
                    first_token_in_entity = False
                else:
                    tags_updates[i] = f'I-{category}'

    return tags_updates


def assign_bio_tags(
    tokens: list[dict],
    entities: list[dict],
    max_workers: int = 8,
    verbose: bool = False,
) -> list[str]:
    """
    Assigns BIO tags to tokens based on entity spans.

    Args:
        tokens: List of tokens with 'text', 'start', and 'end' keys
        entities: List of entities with 'start', 'end', and 'category' keys

    Returns:
        List of BIO tags corresponding to each token
    """

    tags = ['O'] * len(tokens)

    if not entities:
        return tags

    chunk_size = max(1, len(entities) // (max_workers * 4))
    entity_chunks = [
        entities[i : i + chunk_size]
        for i in range(0, len(entities), chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(assign_bio_tags_chunk, tokens, chunk): chunk
            for chunk in entity_chunks
        }

        iterator: Iterator = as_completed(future_to_chunk)
        if verbose:
            iterator = tqdm(
                iterator,
                total=len(entity_chunks),
                desc="Assigning BIO tags",
                unit="chunk",
            )

        for future in iterator:
            try:
                tags_updates = future.result()
                for idx, tag in tags_updates.items():
                    tags[idx] = tag
            except Exception:
                pass

    return tags


def write_bio_format(
    tokens: list[dict], tags: list[str], output_file: str
) -> None:
    """
    Writes tokens and BIO tags to a JSON file.

    Args:
        tokens: List of tokens with 'text', 'start', and 'end' keys
        tags: List of BIO tags corresponding to each token
        output_file: Path to output JSON file
    """
    import json

    training_data = {
        'text': ' '.join(t['text'] for t in tokens),
        'tokens': [t['text'] for t in tokens],
        'tags': tags,
        'token_positions': [
            {'start': t['start'], 'end': t['end']} for t in tokens
        ],
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)


def extract_bio_format(
    original_file: str,
    anon_file: str,
    output_file: str,
    max_workers: int = 8,
    skip_alignment: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Extracts anonymized data and outputs in BIO format.

    Args:
        original_file: Path to original text file
        anon_file: Path to anonymized text file
        output_file: Path to output JSON file
        max_workers: Number of worker threads for parallel processing
        skip_alignment: Skip token alignment method (only use context matching)
        verbose: Show progress bars during processing

    Returns:
        Dictionary with tokens, tags, entities, and number of entities
    """
    original_text = read_file(original_file)
    anon_text = read_file(anon_file)

    pairs1 = extract_pairs(
        original_text, anon_text, max_workers=max_workers, verbose=verbose
    )

    if skip_alignment:
        pairs2 = []
    else:
        pairs2 = align_and_extract(
            original_text, anon_text, max_workers=max_workers, verbose=verbose
        )

    all_pairs = pairs2 + pairs1
    unique_pairs = deduplicate_pairs(all_pairs)

    entities = find_entity_spans(original_text, unique_pairs, verbose=verbose)

    tokens = tokenize_with_positions(original_text, verbose=verbose)

    tags = assign_bio_tags(tokens, entities, verbose=verbose)

    write_bio_format(tokens, tags, output_file)

    return {
        'tokens': [t['text'] for t in tokens],
        'tags': tags,
        'entities': entities,
        'num_entities': len(entities),
    }


extract_bio_format(
    'original.txt',
    'anon.txt',
    'training_data_bio.json',
    verbose=True,
    max_workers=12,
)
