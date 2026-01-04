#!/usr/bin/env python
"""
FlexCTC Evaluation Script for GigaSpeech
Evaluates CTC Batched Beam Search (FlexCTC) with optional N-gram LM fusion.
Uses NeMo's built-in decoding framework.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, List

import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate


def load_manifest(manifest_path: str, max_samples: Optional[int] = None) -> List[dict]:
    """Load manifest file and return list of samples."""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            sample = json.loads(line.strip())
            samples.append(sample)
    return samples


def evaluate_flexctc(
    model_name: str,
    manifest_path: str,
    lm_path: Optional[str] = None,
    lm_alpha: float = 0.5,
    beam_size: int = 16,
    beam_beta: float = 0.3,
    beam_threshold: float = 20.0,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    use_cuda_graphs: bool = True,
    compare_greedy: bool = False,
):
    """
    Evaluate FlexCTC on a manifest file.

    Args:
        model_name: Name of the pretrained CTC model
        manifest_path: Path to the manifest file
        lm_path: Path to the N-gram LM (ARPA format) - optional
        lm_alpha: Weight for LM scores
        beam_size: Beam size for beam search
        beam_beta: Word insertion bonus
        beam_threshold: Beam pruning threshold
        batch_size: Batch size for inference
        max_samples: Maximum number of samples to evaluate (None for all)
        use_cuda_graphs: Whether to use CUDA graphs for acceleration
        compare_greedy: Whether to also run greedy decoding for comparison
    """

    print("=" * 70)
    print("FlexCTC Evaluation")
    print("=" * 70)

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("WARNING: CUDA not available. Running on CPU (slow).")
        use_cuda_graphs = False

    print()

    # Load model
    print(f"[1/4] Loading model: {model_name}")
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)
    model = model.to(device)
    model.eval()

    vocab_size = len(model.decoder.vocabulary) + 1  # +1 for blank
    print(f"      Vocab size: {vocab_size}")
    print()

    # Load manifest and get audio paths + references
    print(f"[2/4] Loading manifest: {manifest_path}")
    samples = load_manifest(manifest_path, max_samples)
    audio_paths = [s['audio_filepath'] for s in samples]
    references = [s['text'] for s in samples]
    durations = [s.get('duration', 0) for s in samples]
    total_audio_duration = sum(durations)
    print(f"      Samples: {len(samples)}")
    print(f"      Total duration: {total_audio_duration:.1f}s ({total_audio_duration/3600:.2f}h)")
    print()

    # Configure FlexCTC decoding (beam_batch strategy)
    print(f"[3/4] Configuring FlexCTC decoder")
    decoding_cfg = OmegaConf.create({
        'strategy': 'beam_batch',
        'beam': {
            'beam_size': beam_size,
            'beam_beta': beam_beta,
            'beam_threshold': beam_threshold,
            'return_best_hypothesis': True,
            'allow_cuda_graphs': use_cuda_graphs,
            'ngram_lm_model': lm_path,
            'ngram_lm_alpha': lm_alpha if lm_path else 0.0,
        }
    })

    model.change_decoding_strategy(decoding_cfg, verbose=False)

    print(f"      Strategy: beam_batch (FlexCTC)")
    print(f"      Beam size: {beam_size}")
    print(f"      Beam beta: {beam_beta}")
    print(f"      CUDA graphs: {use_cuda_graphs}")
    if lm_path:
        print(f"      LM: {lm_path}")
        print(f"      LM alpha: {lm_alpha}")
    else:
        print(f"      LM: None (no language model)")
    print()

    # Run FlexCTC transcription
    print(f"[4/4] Running FlexCTC transcription...")
    start_time = time.time()

    hypotheses_beam_raw = model.transcribe(
        audio=audio_paths,
        batch_size=batch_size,
        return_hypotheses=True,  # Get Hypothesis objects
        verbose=True,
    )
    # Extract text from Hypothesis objects
    hypotheses_beam = [h.text if hasattr(h, 'text') else str(h) for h in hypotheses_beam_raw]

    beam_time = time.time() - start_time

    # Optionally run greedy decoding for comparison
    hypotheses_greedy = None
    greedy_time = 0
    if compare_greedy:
        print()
        print("Running greedy decoding for comparison...")
        greedy_cfg = OmegaConf.create({
            'strategy': 'greedy_batch',
        })
        model.change_decoding_strategy(greedy_cfg, verbose=False)

        start_time = time.time()
        hypotheses_greedy_raw = model.transcribe(
            audio=audio_paths,
            batch_size=batch_size,
            return_hypotheses=True,
            verbose=True,
        )
        # Extract text from Hypothesis objects
        hypotheses_greedy = [h.text if hasattr(h, 'text') else str(h) for h in hypotheses_greedy_raw]
        greedy_time = time.time() - start_time

    print()

    # Calculate WER
    print("=" * 70)
    print("Results")
    print("=" * 70)

    wer_beam = word_error_rate(hypotheses=hypotheses_beam, references=references)
    rtf_beam = beam_time / total_audio_duration if total_audio_duration > 0 else 0
    rtfx_beam = 1.0 / rtf_beam if rtf_beam > 0 else 0

    print()
    print("FlexCTC (beam_batch):")
    print(f"  WER: {wer_beam * 100:.2f}%")
    print(f"  Inference time: {beam_time:.2f}s")
    print(f"  RTF: {rtf_beam:.4f}")
    print(f"  RTFx: {rtfx_beam:.1f}x")

    if hypotheses_greedy:
        wer_greedy = word_error_rate(hypotheses=hypotheses_greedy, references=references)
        rtf_greedy = greedy_time / total_audio_duration if total_audio_duration > 0 else 0
        rtfx_greedy = 1.0 / rtf_greedy if rtf_greedy > 0 else 0

        print()
        print("Greedy (baseline):")
        print(f"  WER: {wer_greedy * 100:.2f}%")
        print(f"  Inference time: {greedy_time:.2f}s")
        print(f"  RTF: {rtf_greedy:.4f}")
        print(f"  RTFx: {rtfx_greedy:.1f}x")

        print()
        print(f"WER improvement: {(wer_greedy - wer_beam) * 100:.2f}% absolute")

    print()

    # Show some examples
    print("Sample predictions:")
    print("-" * 70)
    for i in range(min(5, len(hypotheses_beam))):
        print(f"REF: {references[i]}")
        print(f"HYP: {hypotheses_beam[i]}")
        if hypotheses_greedy:
            print(f"GRD: {hypotheses_greedy[i]}")
        print()

    return wer_beam


def main():
    parser = argparse.ArgumentParser(description="FlexCTC Evaluation Script")
    parser.add_argument(
        "--model",
        type=str,
        default="stt_en_conformer_ctc_large",
        help="Pretrained model name"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to test manifest file"
    )
    parser.add_argument(
        "--lm",
        type=str,
        default=None,
        help="Path to N-gram LM (ARPA format, not KenLM binary)"
    )
    parser.add_argument(
        "--lm_alpha",
        type=float,
        default=0.5,
        help="LM weight"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=16,
        help="Beam size"
    )
    parser.add_argument(
        "--beam_beta",
        type=float,
        default=0.3,
        help="Word insertion bonus"
    )
    parser.add_argument(
        "--beam_threshold",
        type=float,
        default=20.0,
        help="Beam pruning threshold"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--no_cuda_graphs",
        action="store_true",
        help="Disable CUDA graphs"
    )
    parser.add_argument(
        "--compare_greedy",
        action="store_true",
        help="Also run greedy decoding for comparison"
    )

    args = parser.parse_args()

    evaluate_flexctc(
        model_name=args.model,
        manifest_path=args.manifest,
        lm_path=args.lm,
        lm_alpha=args.lm_alpha,
        beam_size=args.beam_size,
        beam_beta=args.beam_beta,
        beam_threshold=args.beam_threshold,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        use_cuda_graphs=not args.no_cuda_graphs,
        compare_greedy=args.compare_greedy,
    )


if __name__ == "__main__":
    main()
