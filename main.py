#!/usr/bin/env python3
"""
Demonstracja biblioteki do anonimizacji tekstu za pomocą regex.

Pobiera losowe 100 linii z plików treningowych NASK i porównuje:
- original.txt - oryginalne teksty z danymi osobowymi
- anon.txt - teksty po anonimizacji (referencja)

Pokazuje jak działa nasz anonymizer i porównuje z oczekiwanym wynikiem.
"""

import random
import os
from typing import List, Tuple
from regex import RegexAnonymizer, anonymize_text


def load_data(directory: str = "nask_train") -> Tuple[List[str], List[str]]:
    """
    Wczytuje dane treningowe.
    
    Returns:
        Tuple (original_lines, anon_lines) - listy linii z obu plików
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, directory)
    
    original_path = os.path.join(data_dir, "original.txt")
    anon_path = os.path.join(data_dir, "anon.txt")
    
    with open(original_path, 'r', encoding='utf-8') as f:
        original_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(anon_path, 'r', encoding='utf-8') as f:
        anon_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    return original_lines, anon_lines


def get_sample_pairs(original: List[str], anon: List[str], 
                     n: int = 100, seed: int = 42) -> List[Tuple[int, str, str]]:
    """
    Pobiera n losowych par (indeks, original, anon).
    
    Args:
        original: Lista oryginalnych linii
        anon: Lista zanonimizowanych linii
        n: Liczba par do pobrania
        seed: Seed dla generatora losowego (dla powtarzalności)
    
    Returns:
        Lista krotek (indeks, oryginalna linia, zanonimizowana linia)
    """
    random.seed(seed)
    
    # Upewnij się, że obie listy mają tę samą długość
    min_len = min(len(original), len(anon))
    indices = random.sample(range(min_len), min(n, min_len))
    
    return [(i, original[i], anon[i]) for i in sorted(indices)]


def highlight_differences(original: str, anonymized: str, reference: str) -> dict:
    """
    Porównuje wynik anonimizacji z referencją.
    
    Returns:
        dict z informacjami o różnicach
    """
    # Sprawdź czy nasze tagi zostały zastąpione
    our_tags = ['[pesel]', '[email]', '[phone]', '[bank-account]', 
                '[credit-card-number]', '[document-number]']
    
    ref_tags = ['[pesel]', '[email]', '[phone]', '[bank-account]',
                '[credit-card-number]', '[document-number]',
                '[name]', '[surname]', '[city]', '[address]',
                '[age]', '[sex]', '[date]', '[company]', '[relative]', '[health]']
    
    our_found = sum(1 for tag in our_tags if tag in anonymized)
    ref_found = sum(1 for tag in ref_tags if tag in reference)
    
    # Zlicz ile podstawowych tagów (które obsługujemy) jest w referencji
    regex_tags_in_ref = sum(1 for tag in our_tags if tag in reference)
    
    return {
        'our_tags_used': our_found,
        'ref_tags_used': ref_found,
        'regex_tags_in_ref': regex_tags_in_ref,
        'match_ratio': our_found / max(regex_tags_in_ref, 1) if regex_tags_in_ref > 0 else 1.0
    }


def demo_single_line(anonymizer: RegexAnonymizer, line: str, show_details: bool = True):
    """
    Demonstracja anonimizacji pojedynczej linii.
    """
    result = anonymizer.anonymize_detailed(line)
    
    print("=" * 80)
    print("ORYGINALNY TEKST:")
    print("-" * 40)
    print(line[:500] + ("..." if len(line) > 500 else ""))
    print()
    print("ZANONIMIZOWANY:")
    print("-" * 40)
    print(result.anonymized_text[:500] + ("..." if len(result.anonymized_text) > 500 else ""))
    
    if show_details and result.replacements:
        print()
        print("ZNALEZIONE DANE OSOBOWE:")
        print("-" * 40)
        for original_val, tag, pos in result.replacements[:10]:  # Max 10
            print(f"  • [{tag}] '{original_val}'")
        if len(result.replacements) > 10:
            print(f"  ... i {len(result.replacements) - 10} więcej")
    print()


def main():
    """Główna funkcja demonstracyjna."""
    
    print("=" * 80)
    print("  DEMONSTRACJA BIBLIOTEKI REGEX ANONYMIZER")
    print("=" * 80)
    print()
    
    # Wczytaj dane
    print("📂 Wczytywanie danych treningowych...")
    try:
        original_lines, anon_lines = load_data()
        print(f"   Wczytano {len(original_lines)} linii z original.txt")
        print(f"   Wczytano {len(anon_lines)} linii z anon.txt")
    except FileNotFoundError as e:
        print(f"❌ Błąd: Nie znaleziono plików treningowych: {e}")
        return
    
    print()
    
    # Inicjalizuj anonymizer
    anonymizer = RegexAnonymizer(use_brackets=True)
    
    # Pobierz 100 losowych par
    print("🎲 Pobieram 100 losowych linii do analizy...")
    pairs = get_sample_pairs(original_lines, anon_lines, n=100)
    print(f"   Pobrano {len(pairs)} par")
    print()
    
    # Statystyki
    total_our_tags = 0
    total_ref_regex_tags = 0
    matches = 0
    
    # Pokazuj szczegóły dla pierwszych 5 linii
    print("📊 PRZYKŁADY ANONIMIZACJI (pierwsze 5):")
    print("=" * 80)
    
    for i, (idx, original, reference) in enumerate(pairs[:5]):
        print(f"\n--- Linia {idx + 1} ---")
        
        # Nasza anonimizacja
        our_result = anonymizer.anonymize(original)
        
        print("ORYGINAŁ (fragment):")
        print(original[:300] + ("..." if len(original) > 300 else ""))
        print()
        print("NASZA ANONIMIZACJA (fragment):")
        print(our_result[:300] + ("..." if len(our_result) > 300 else ""))
        print()
        print("REFERENCYJNA ANONIMIZACJA (fragment):")
        print(reference[:300] + ("..." if len(reference) > 300 else ""))
        print()
        
        # Porównanie
        stats = highlight_differences(original, our_result, reference)
        print(f"📈 Statystyki: Nasze tagi: {stats['our_tags_used']}, "
              f"Tagi regex w referencji: {stats['regex_tags_in_ref']}, "
              f"Match ratio: {stats['match_ratio']:.2%}")
    
    # Analiza wszystkich 100 linii
    print()
    print("=" * 80)
    print("📊 PODSUMOWANIE ANALIZY 100 LINII")
    print("=" * 80)
    
    results = []
    for idx, original, reference in pairs:
        our_result = anonymizer.anonymize(original)
        stats = highlight_differences(original, our_result, reference)
        results.append({
            'idx': idx,
            'original': original,
            'our_result': our_result,
            'reference': reference,
            **stats
        })
        total_our_tags += stats['our_tags_used']
        total_ref_regex_tags += stats['regex_tags_in_ref']
        if stats['our_tags_used'] > 0 and stats['our_tags_used'] >= stats['regex_tags_in_ref']:
            matches += 1
    
    print()
    print(f"📈 STATYSTYKI ZBIORCZE:")
    print(f"   • Łączna liczba naszych tagów: {total_our_tags}")
    print(f"   • Łączna liczba tagów regex w referencji: {total_ref_regex_tags}")
    print(f"   • Linie z co najmniej 1 znalezionym tagiem regex: {sum(1 for r in results if r['our_tags_used'] > 0)}/100")
    
    # Znajdź przykłady gdzie znaleźliśmy dużo
    good_examples = [r for r in results if r['our_tags_used'] >= 3]
    if good_examples:
        print()
        print(f"🎯 PRZYKŁAD Z WIELOMA WYKRYTYMI DANYMI (znaleziono {good_examples[0]['our_tags_used']} tagów):")
        print("-" * 60)
        print("Oryginał (fragment):")
        print(good_examples[0]['original'][:400])
        print()
        print("Nasza anonimizacja (fragment):")
        print(good_examples[0]['our_result'][:400])
    
    # Pokaż jakie tagi obsługujemy
    print()
    print("=" * 80)
    print("ℹ️  OBSŁUGIWANE TYPY DANYCH:")
    print("=" * 80)
    print("""
    • [pesel]              - PESEL (11 cyfr)
    • [email]              - Adresy e-mail
    • [phone]              - Numery telefonów (różne formaty polskie)
    • [bank-account]       - Numery kont bankowych (IBAN)
    • [credit-card-number] - Numery kart kredytowych (16 cyfr)
    • [document-number]    - Numery dowodów osobistych
    
    ⚠️  NIEOBSŁUGIWANE (wymagają NLP/ML):
    • [name], [surname]    - Imiona i nazwiska
    • [city], [address]    - Miasta i adresy
    • [age], [sex]         - Wiek i płeć
    • [date]               - Daty
    • [company]            - Nazwy firm
    """)
    
    print()
    print("✅ Demonstracja zakończona!")
    print()


if __name__ == "__main__":
    main()
