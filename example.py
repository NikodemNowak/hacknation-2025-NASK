#!/usr/bin/env python3
"""
Example usage of the anonymizer library.

Run this script to see how the anonymization works:
    python example.py
"""

from anonymizer import Anonymizer

# Example texts with various types of PII
example_texts = [
    "Nazywam się Jan Kowalski, mieszkam w Warszawie.",
    "Mój PESEL to 90010112345, a email: jan.kowalski@example.com",
    "Zadzwoń pod numer +48 123 456 789 lub napisz na adam@firma.pl",
    "Numer konta: PL61 1090 1014 0000 0712 1981 2874",
    "Pracuję w firmie Google jako programista.",
]

if __name__ == "__main__":
    print("=" * 60)
    print("Anonymizer - Example Usage")
    print("=" * 60)
    print()
    
    # Initialize the anonymizer
    # On first run, the NER model will be downloaded from HuggingFace
    print("Initializing anonymizer (model will download if not cached)...")
    anon = Anonymizer()
    print(f"Anonymizer ready: {anon}")
    print()
    
    # Process each example
    for i, text in enumerate(example_texts, 1):
        print(f"--- Example {i} ---")
        print(f"Original:   {text}")
        
        result = anon.anonymize(text)
        print(f"Anonymized: {result}")
        print()
    
    # Batch processing
    print("--- Batch Processing ---")
    results = anon.anonymize_batch(example_texts)
    print(f"Processed {len(results)} texts in batch.")
    print()
    
    # Show supported tags
    print("--- Supported Tags ---")
    tags = anon.get_supported_tags()
    print(f"Total: {len(tags)} tags")
    print(", ".join(tags[:10]) + "...")  # Show first 10
