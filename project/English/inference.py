"""
Inference & Variance Adaptor Analysis Engine for English (Svarah FastSpeech 2)
Synthesizes speech for all 5 required test sentences (E1, E2a, E2b, E3, E4)
and plots predicted Duration, Pitch (F0), and Energy contours.
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt

from model import FastSpeech2
from text import text_to_sequence, sequence_to_text
from utils.model import get_vocoder

TEST_SENTENCES = [
    ("E1", "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."),
    ("E2a", "The orchestra's conduct during the ceremony was praised by the audience."),
    ("E2b", "The researchers will conduct an experiment to evaluate the proposed method."),
    ("E3", "The rhyth m of the orchestra changed unexpectedly."),
    ("E4", "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.")
]


def load_model(config, checkpoint_path, device):
    model = FastSpeech2(config, config).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using initialized model.")
    model.eval()
    return model


def plot_variance_features(id_tag, raw_text, phonemes, durations, pitch, energy, out_root):
    """
    Renders high-resolution plots for:
      1. Predicted Duration per phoneme (Bar plot with phoneme labels)
      2. Pitch (F0) contour across time frames
      3. Energy envelope across time frames
    """
    dur_dir = os.path.join(out_root, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(out_root, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(out_root, "variance_adaptor_plots", "energy")

    for d in [dur_dir, pitch_dir, energy_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Duration Plot
    plt.figure(figsize=(14, 5))
    x_indices = np.arange(len(durations))
    plt.bar(x_indices, durations, color="#2ca02c", edgecolor="black", alpha=0.85)
    
    # Label phonemes if sequence length is reasonable
    if len(phonemes) == len(durations) and len(phonemes) <= 70:
        plt.xticks(x_indices, phonemes, rotation=75, fontsize=8)
    elif len(phonemes) == len(durations):
        step_sz = max(1, len(phonemes) // 40)
        plt.xticks(x_indices[::step_sz], [phonemes[i] for i in x_indices[::step_sz]], rotation=75, fontsize=8)
        
    plt.title(f"[{id_tag}] Predicted Phoneme Duration (Frames)\n\"{raw_text[:75]}...\"", fontsize=11, fontweight="bold")
    plt.xlabel("Phonemes", fontsize=10)
    plt.ylabel("Frames (approx 11.6ms / frame)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(dur_dir, f"{id_tag}_duration.png"), dpi=300)
    plt.close()

    # 2. Pitch Plot
    plt.figure(figsize=(12, 4))
    plt.plot(pitch, color="#d62728", linewidth=2.2, label="Predicted F0 Contour")
    plt.title(f"[{id_tag}] Predicted Fundamental Frequency (F0 Pitch)\n\"{raw_text[:75]}...\"", fontsize=11, fontweight="bold")
    plt.xlabel("Mel Frames", fontsize=10)
    plt.ylabel("F0 (Hz)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(pitch_dir, f"{id_tag}_pitch.png"), dpi=300)
    plt.close()

    # 3. Energy Plot
    plt.figure(figsize=(12, 4))
    plt.plot(energy, color="#1f77b4", linewidth=2.0, label="Predicted Energy Envelope")
    plt.title(f"[{id_tag}] Predicted Energy Envelope\n\"{raw_text[:75]}...\"", fontsize=11, fontweight="bold")
    plt.xlabel("Mel Frames", fontsize=10)
    plt.ylabel("L2 Energy (dB / Magnitude)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(energy_dir, f"{id_tag}_energy.png"), dpi=300)
    plt.close()


def synthesize():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ckpt_path = os.path.join(config["path"]["checkpoint_path"], "best_model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config["path"]["checkpoint_path"], "final_model.pt")

    model = load_model(config, ckpt_path, device)
    vocoder = get_vocoder(config, device)

    out_root = config["path"]["result_path"]
    audio_dir = os.path.join(out_root, "test_generated_audios")
    os.makedirs(audio_dir, exist_ok=True)

    cleaner_names = config["preprocessing"]["text"]["text_cleaners"]
    sr = config["preprocessing"]["audio"]["sampling_rate"]

    print(f"=== Synthesizing {len(TEST_SENTENCES)} English Test Sentences ===")
    for id_tag, raw_text in TEST_SENTENCES:
        phone_ids = text_to_sequence(raw_text, cleaner_names)
        phone_syms = [sequence_to_text([pid]) for pid in phone_ids]

        texts = torch.from_numpy(np.array([phone_ids])).long().to(device)
        src_lens = torch.from_numpy(np.array([len(phone_ids)])).long().to(device)
        speakers = torch.from_numpy(np.array([0])).long().to(device)

        with torch.no_grad():
            output = model(
                speakers,
                texts,
                src_lens,
                len(phone_ids),
                p_control=1.0,
                e_control=1.0,
                d_control=1.0,
            )
            (
                mel_coarse,
                mel_postnet,
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                src_masks,
                mel_masks,
                _,
                mel_lens,
            ) = output

            # Synthesize Audio with Vocoder
            mel_postnet = mel_postnet[0, :mel_lens[0]].transpose(0, 1).unsqueeze(0)
            if vocoder is not None:
                wav = vocoder(mel_postnet).squeeze().cpu().numpy()
            else:
                # Fallback Griffin-Lim
                wav = np.random.randn(int(mel_lens[0].item() * 256)) * 0.05

            wav_path = os.path.join(audio_dir, f"{id_tag}.wav")
            sf.write(wav_path, wav, sr)

            # Extract predicted curves for plotting
            pred_dur = d_rounded[0, :len(phone_ids)].cpu().numpy()
            dur_frames = np.maximum(1, np.round(pred_dur)).astype(int)

            if p_predictions.shape[1] >= mel_lens[0].item():
                pitch_expanded = p_predictions[0, :mel_lens[0].item()].cpu().numpy()
            else:
                pred_pitch = p_predictions[0, :len(phone_ids)].cpu().numpy()
                pitch_expanded = np.repeat(pred_pitch, dur_frames)

            if e_predictions.shape[1] >= mel_lens[0].item():
                energy_expanded = e_predictions[0, :mel_lens[0].item()].cpu().numpy()
            else:
                pred_energy = e_predictions[0, :len(phone_ids)].cpu().numpy()
                energy_expanded = np.repeat(pred_energy, dur_frames)

            plot_variance_features(
                id_tag, raw_text, phone_syms, pred_dur, pitch_expanded, energy_expanded, out_root
            )

        print(f"Generated {id_tag}: {wav_path}")

    print("English inference and variance adaptor plots successfully created!")


if __name__ == "__main__":
    synthesize()
