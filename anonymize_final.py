#!/usr/bin/env python3
"""
Wsadowa anonimizacja pliku dane_final_test/orig_final.txt przy użyciu
pełnego Anonymizer z anonymizer/core.py (RegEx + NER, bez syntezy).

- Czyta linie z orig_final.txt (jedna linia = jeden tekst).
- Anonimizuje każdą linię Anonymizerem (RegEx + NER).
- Zapisuje wynik do dane_final_test/anonymized_final.txt w tej samej kolejności.
- Mierzy czas działania i wypisuje go na stdout.
- Opcjonalnie (flaga --log-performance) dopisze wynik do performance_all_in.txt.
"""

import argparse
import time
from pathlib import Path

from anonymizer import Anonymizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anonimizacja orig_final.txt -> anonymized_final.txt (RegEx + NER)."
    )
    parser.add_argument(
        "--log-performance",
        action="store_true",
        help="Dopisz wynik do performance_all_in.txt (na końcu pliku).",
    )
    parser.add_argument(
        "--brackets",
        action="store_true",
        help="Użyj nawiasów kwadratowych [tag] zamiast klamrowych {tag}.",
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Wyłącz NER (zostanie tylko warstwa RegEx).",
    )
    parser.add_argument(
        "--no-regex",
        action="store_true",
        help="Wyłącz RegEx (zostanie tylko warstwa NER).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    input_path = repo_root / "dane_final_test" / "orig_final.txt"
    output_path = input_path.with_name("anonymized_final.txt")
    performance_path = repo_root / "performance_all_in.txt"

    if not input_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego: {input_path}")

    start = time.perf_counter()

    anonymizer = Anonymizer(
        use_regex=not args.no_regex,
        use_ner=not args.no_ner,
        use_brackets=args.brackets,
        use_synthetic=False,
    )
    lines = input_path.read_text(encoding="utf-8").splitlines()
    anonymized_lines = [anonymizer.anonymize(line) for line in lines]

    output_path.write_text("\n".join(anonymized_lines) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - start

    print(f"Zakończono. Linie: {len(anonymized_lines)}.")
    print(f"Plik wyjściowy: {output_path.relative_to(repo_root)}")
    print(f"Czas: {elapsed:.2f} s")

    if args.log_performance:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        entry = (
            f"\nRun: {timestamp} UTC\n"
            f"  Wejście: {input_path.relative_to(repo_root)}\n"
            f"  Wyjście: {output_path.relative_to(repo_root)}\n"
            f"  Linie: {len(anonymized_lines)}\n"
            f"  Czas: {elapsed:.2f} s\n"
            f"  Warstwy: RegEx={'ON' if not args.no_regex else 'OFF'}, "
            f"NER={'ON' if not args.no_ner else 'OFF'} (syntetyczna: OFF)\n"
        )
        with performance_path.open("a", encoding="utf-8") as f:
            f.write(entry)


if __name__ == "__main__":
    main()
