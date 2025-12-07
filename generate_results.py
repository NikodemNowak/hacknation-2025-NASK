import time
import platform
import os
from anonymizer import Anonymizer

def get_cpu_info():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except:
        return platform.processor()

def main():
    start_time = time.time()
    
    # Load data
    with open("nask_train/original.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()] # Keep empty lines? PDF says "Nie usuwajcie zdań!". It also says "Liczba linii... identyczna". 
        # main.py does [line.strip() for line in f.readlines() if line.strip()]. I should probably NOT strip empty lines if they exist, or at least keep the count.
        # Let's re-read original file without stripping to be safe, but strip newline char.

    with open("nask_train/original.txt", "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    # Initialize Anonymizer
    # Use brackets as required, and point to local model
    model = Anonymizer(use_brackets=True, ner_model_path="models/herbert_ner_v2")
    
    # Process
    anonymized_lines = []
    for line in raw_lines:
        if not line.strip():
            anonymized_lines.append("")
            continue
        # anonymize
        result = model.anonymize(line)
        anonymized_lines.append(result)
        
    end_time = time.time()
    duration = end_time - start_time
    
    # Write output
    with open("output_all_in.txt", "w", encoding="utf-8") as f:
        for line in anonymized_lines:
            f.write(line + "\n")
            
    # Write performance
    cpu_name = get_cpu_info()
    performance_content = f"""Time: {duration:.2f} s
Hardware:
  Typ: CPU
  Model: {cpu_name}
API:
  Czy użyto API PLLUM: NIE
  Model: HerBERT (local) + Regex
"""
    with open("performance_all_in.txt", "w", encoding="utf-8") as f:
        f.write(performance_content)

if __name__ == "__main__":
    main()
