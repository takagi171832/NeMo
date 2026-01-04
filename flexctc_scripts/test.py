#!/usr/bin/env python
"""
FlexCTC Test Script
Tests the CTC Batched Beam Search (FlexCTC) functionality with a pre-trained model.
"""

import torch
import numpy as np

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.ctc_batched_beam_decoding import BatchedBeamCTCComputer
from nemo.collections.asr.parts.utils.batched_beam_decoding_utils import BatchedBeamHyps


def test_flexctc():
    print("=" * 60)
    print("FlexCTC Test Script")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. FlexCTC requires GPU.")
        print("Running in CPU mode for basic testing only.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print()

    # Step 1: Load a pre-trained CTC model
    print("[1/4] Loading pre-trained CTC model...")
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="stt_en_conformer_ctc_small"
    )
    model = model.to(device)
    model.eval()
    print(f"      Model loaded: stt_en_conformer_ctc_small")
    print(f"      Vocab size: {model.decoder.vocabulary.__len__() + 1}")  # +1 for blank
    print()

    # Step 2: Create synthetic input for testing
    print("[2/4] Creating synthetic test input...")
    batch_size = 2
    seq_len = 100
    vocab_size = len(model.decoder.vocabulary) + 1  # +1 for blank

    # Create random log probabilities (simulating encoder output after decoder projection)
    log_probs = torch.randn(batch_size, seq_len, vocab_size, device=device)
    log_probs = torch.log_softmax(log_probs, dim=-1)
    lengths = torch.tensor([seq_len, seq_len // 2], device=device)

    print(f"      Batch size: {batch_size}")
    print(f"      Sequence length: {seq_len}")
    print(f"      Vocab size: {vocab_size}")
    print()

    # Step 3: Test BatchedBeamCTCComputer (FlexCTC core)
    print("[3/4] Testing BatchedBeamCTCComputer (FlexCTC)...")

    blank_index = vocab_size - 1  # Typically blank is at the end
    beam_size = 4

    beam_computer = BatchedBeamCTCComputer(
        blank_index=blank_index,
        beam_size=beam_size,
        return_best_hypothesis=True,
        allow_cuda_graphs=device == "cuda",
    )

    print(f"      Beam size: {beam_size}")
    print(f"      CUDA graphs mode: {beam_computer.cuda_graphs_mode}")

    # Run beam search
    with torch.no_grad():
        batched_hyps = beam_computer(log_probs, lengths)

    print(f"      Beam search completed successfully!")
    print(f"      Number of hypotheses: {batched_hyps.scores.shape}")
    print()

    # Step 4: Decode results
    print("[4/4] Decoding results...")

    # Convert to list of hypotheses using the proper API
    hypotheses = batched_hyps.to_hyps_list(score_norm=True)

    for batch_idx, hyp in enumerate(hypotheses):
        print(f"      Batch {batch_idx}:")
        print(f"        Score: {hyp.score:.4f}")
        tokens = hyp.y_sequence.tolist()
        # Filter blank tokens
        tokens = [t for t in tokens if t != blank_index]
        if tokens:
            text = model.tokenizer.ids_to_text(tokens)
            print(f"        Text: '{text}'")
            print(f"        Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        else:
            print(f"        Text: (empty)")

    print()
    print("=" * 60)
    print("FlexCTC test completed successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_flexctc()
    exit(0 if success else 1)
