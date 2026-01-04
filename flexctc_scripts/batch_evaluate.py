#!/usr/bin/env python
"""
FlexCTC Evaluation Script for All GigaSpeech Categories
"""

import subprocess
import sys
from pathlib import Path

# Configuration
CATEGORIES = [0, 6, 12, 17, 21, 28]
CATEGORY_NAMES = ["y-people", "p-news", "y-news", "y-science", "y-education", "audiobook"]
MANIFEST_BASE = "/home/takagi/NeMo_arxiv/manifests/GigaSpeech"
LM_BASE = "/home/takagi/NeMo/models/LM/GigaSpeech_FlexCTC"
RESULTS_DIR = "/home/takagi/NeMo/results/FlexCTC"
MODEL = "stt_en_conformer_ctc_large_ls"
BEAM_SIZE = 20
LM_ALPHA = 0.5
BATCH_SIZE = 32

# Create results directory
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

results_summary = []

print("=" * 70)
print("FlexCTC Evaluation - All Categories")
print("=" * 70)
print(f"Model: {MODEL}")
print(f"Beam size: {BEAM_SIZE}")
print(f"LM alpha: {LM_ALPHA}")
print(f"Batch size: {BATCH_SIZE}")
print("=" * 70)
print()

for i, category in enumerate(CATEGORIES):
    category_name = CATEGORY_NAMES[i]

    test_manifest = f"{MANIFEST_BASE}/category_{category}/split/test_manifest.json"
    lm_file = f"{LM_BASE}/category_{category}_4gram.bin.nemo"
    log_file = f"{RESULTS_DIR}/category_{category}_{category_name}.log"

    print(f"\n[{i+1}/6] Evaluating Category {category} ({category_name})")
    print(f"  Manifest: {test_manifest}")
    print(f"  LM: {lm_file}")
    print("-" * 70)

    cmd = [
        sys.executable,
        "evaluate.py",
        "--model", MODEL,
        "--manifest", test_manifest,
        "--lm", lm_file,
        "--lm_alpha", str(LM_ALPHA),
        "--beam_size", str(BEAM_SIZE),
        "--batch_size", str(BATCH_SIZE),
        "--compare_greedy",
    ]

    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout
        f.write(output)
        print(output)

    # Extract WER from output
    for line in output.split('\n'):
        if 'FlexCTC (beam_batch):' in line:
            results_summary.append(f"\nCategory {category} ({category_name}):")
        elif 'WER:' in line or 'RTFx:' in line:
            results_summary.append(f"  {line.strip()}")

    print(f"  Log saved to: {log_file}")

# Save summary
summary_file = f"{RESULTS_DIR}/summary.txt"
with open(summary_file, 'w') as f:
    f.write("FlexCTC Evaluation Summary\n")
    f.write("=" * 70 + "\n")
    f.write(f"Model: {MODEL}\n")
    f.write(f"Beam size: {BEAM_SIZE}\n")
    f.write(f"LM alpha: {LM_ALPHA}\n")
    f.write("\n")
    f.write("\n".join(results_summary))

print("\n" + "=" * 70)
print("All evaluations completed!")
print("=" * 70)
print("\nSummary:")
print("\n".join(results_summary))
print(f"\nFull summary saved to: {summary_file}")
