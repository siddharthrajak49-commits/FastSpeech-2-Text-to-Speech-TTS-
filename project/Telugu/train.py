"""
FastSpeech 2 Training Engine for Telugu (IndicTTS Telugu Dataset)
Includes scheduled learning rate, gradient clipping, validation monitoring,
checkpointing, and automatic training/validation loss curve generation.
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model import FastSpeech2, FastSpeech2Loss
from utils.model import get_param_num
from utils.tools import pad_1D, pad_2D


class FastSpeechDataset(Dataset):
    def __init__(self, filename, config):
        self.preprocessed_path = config["path"]["preprocessed_path"]
        self.cleaners = config["preprocessing"]["text"]["text_cleaners"]

        with open(os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8") as f:
            self.lines = [line.strip().split("|") for line in f if len(line.strip()) > 0]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        basename = line[0]
        speaker = 0
        phone_ids = np.array([int(p) for p in line[2].split()])
        raw_text = line[3]

        mel = np.load(os.path.join(self.preprocessed_path, "mel", f"{basename}.npy"))
        pitch = np.load(os.path.join(self.preprocessed_path, "pitch", f"{basename}.npy"))
        energy = np.load(os.path.join(self.preprocessed_path, "energy", f"{basename}.npy"))
        duration = np.load(os.path.join(self.preprocessed_path, "duration", f"{basename}.npy"))

        return {
            "id": basename,
            "speaker": speaker,
            "text": phone_ids,
            "raw_text": raw_text,
            "mel": mel.T,  # shape: (n_frames, n_mels)
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }


def collate_fn(batch):
    ids = [b["id"] for b in batch]
    speakers = np.array([b["speaker"] for b in batch])
    texts = [b["text"] for b in batch]
    raw_texts = [b["raw_text"] for b in batch]
    mels = [b["mel"] for b in batch]
    pitches = [b["pitch"] for b in batch]
    energies = [b["energy"] for b in batch]
    durations = [b["duration"] for b in batch]

    text_lens = np.array([len(t) for t in texts])
    mel_lens = np.array([m.shape[0] for m in mels])

    texts_padded = pad_1D(texts)
    mels_padded = pad_2D(mels)
    pitches_padded = pad_1D(pitches)
    energies_padded = pad_1D(energies)
    durations_padded = pad_1D(durations)

    return (
        ids,
        raw_texts,
        torch.from_numpy(speakers).long(),
        torch.from_numpy(texts_padded).long(),
        torch.from_numpy(text_lens).long(),
        int(max(text_lens)),
        torch.from_numpy(mels_padded).float(),
        torch.from_numpy(mel_lens).long(),
        int(max(mel_lens)),
        torch.from_numpy(pitches_padded).float(),
        torch.from_numpy(energies_padded).float(),
        torch.from_numpy(durations_padded).long(),
    )


def plot_loss_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    steps = history["steps"]

    # 1. Training Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, history["train_total"], label="Total Loss", color="#1f77b4", linewidth=2)
    plt.plot(steps, history["train_mel"], label="Mel Loss", color="#ff7f0e", linestyle="--")
    plt.plot(steps, history["train_dur"], label="Duration Loss", color="#2ca02c", linestyle=":")
    plt.plot(steps, history["train_pitch"], label="Pitch Loss", color="#d62728", linestyle="-.")
    plt.plot(steps, history["train_energy"], label="Energy Loss", color="#9467bd", linestyle="-.")
    plt.title("Telugu FastSpeech 2 Training Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300)
    plt.close()

    # 2. Validation Loss Plot
    if len(history["val_steps"]) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(history["val_steps"], history["val_total"], label="Validation Total Loss", color="#d62728", linewidth=2)
        plt.plot(history["val_steps"], history["val_mel"], label="Validation Mel Loss", color="#ff7f0e", linestyle="--")
        plt.title("Telugu FastSpeech 2 Validation Loss", fontsize=14, fontweight="bold")
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "validation_loss.png"), dpi=300)
        plt.close()


def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training on device: {device}")

    # Datasets
    train_dataset = FastSpeechDataset("train.txt", config)
    val_dataset = FastSpeechDataset("val.txt", config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Model & Loss
    model = FastSpeech2(config, config).to(device)
    loss_fn = FastSpeech2Loss(config, config)
    print(f"Telugu FastSpeech 2 Parameters: {get_param_num(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        betas=config["train"]["betas"],
        eps=config["train"]["eps"],
        weight_decay=config["train"]["weight_decay"]
    )

    ckpt_dir = config["path"]["checkpoint_path"]
    results_dir = config["path"]["result_path"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    history = {
        "steps": [],
        "train_total": [],
        "train_mel": [],
        "train_dur": [],
        "train_pitch": [],
        "train_energy": [],
        "val_steps": [],
        "val_total": [],
        "val_mel": []
    }

    step = 0
    epochs = config["train"]["epochs"]
    best_val_loss = float("inf")

    print(f"Beginning Telugu training for {epochs} epochs...")
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                step += 1
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                (
                    _,
                    _,
                    speakers,
                    texts,
                    src_lens,
                    max_src_len,
                    mels,
                    mel_lens,
                    max_mel_len,
                    p_targets,
                    e_targets,
                    d_targets,
                ) = batch

                optimizer.zero_grad()
                output = model(
                    speakers,
                    texts,
                    src_lens,
                    max_src_len,
                    mels,
                    mel_lens,
                    max_mel_len,
                    p_targets,
                    e_targets,
                    d_targets,
                )

                total_loss, mel_loss, postnet_loss, p_loss, e_loss, d_loss = loss_fn(
                    batch[2:], output
                )

                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_thresh"])
                optimizer.step()

                if step % config["train"]["log_step"] == 0:
                    print(
                        f"Epoch [{epoch}/{epochs}] Step [{step}]: "
                        f"Total Loss: {total_loss.item():.4f} | "
                        f"Mel: {mel_loss.item():.4f} | "
                        f"Dur: {d_loss.item():.4f} | "
                        f"Pitch: {p_loss.item():.4f} | "
                        f"Energy: {e_loss.item():.4f}"
                    )
                    history["steps"].append(step)
                    history["train_total"].append(total_loss.item())
                    history["train_mel"].append(mel_loss.item())
                    history["train_dur"].append(d_loss.item())
                    history["train_pitch"].append(p_loss.item())
                    history["train_energy"].append(e_loss.item())

                # Validation
                if step % config["train"]["val_step"] == 0:
                    model.eval()
                    val_losses = []
                    val_mel_losses = []
                    with torch.no_grad():
                        for v_batch in val_loader:
                            v_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in v_batch]
                            (
                                _,
                                _,
                                v_spk,
                                v_txt,
                                v_slen,
                                v_mslen,
                                v_mel,
                                v_mlen,
                                v_mmlen,
                                v_pt,
                                v_et,
                                v_dt,
                            ) = v_batch

                            v_out = model(
                                v_spk,
                                v_txt,
                                v_slen,
                                v_mslen,
                                v_mel,
                                v_mlen,
                                v_mmlen,
                                v_pt,
                                v_et,
                                v_dt,
                            )
                            v_total, v_mel_l, _, _, _, _ = loss_fn(v_batch[2:], v_out)
                            val_losses.append(v_total.item())
                            val_mel_losses.append(v_mel_l.item())

                    avg_val_loss = float(np.mean(val_losses))
                    avg_val_mel = float(np.mean(val_mel_losses))
                    print(f"-- Validation Loss at Step {step}: {avg_val_loss:.4f} (Mel: {avg_val_mel:.4f}) --")

                    history["val_steps"].append(step)
                    history["val_total"].append(avg_val_loss)
                    history["val_mel"].append(avg_val_mel)

                    plot_loss_curves(history, results_dir)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(
                            {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                            os.path.join(ckpt_dir, "best_model.pt"),
                        )

                    model.train()

                # Save regular checkpoint
                if step % config["train"]["save_step"] == 0:
                    torch.save(
                        {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                        os.path.join(ckpt_dir, f"checkpoint_{step}.pt"),
                    )
    except KeyboardInterrupt:
        print(f"\nTelugu training gracefully paused by user at step {step}.")
    finally:
        if step > 0:
            torch.save(
                {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                os.path.join(ckpt_dir, "final_model.pt"),
            )
            if not os.path.exists(os.path.join(ckpt_dir, "best_model.pt")):
                torch.save(
                    {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    os.path.join(ckpt_dir, "best_model.pt"),
                )
            if len(history["steps"]) > 0:
                plot_loss_curves(history, results_dir)
            print(f"Latest Telugu model checkpoint & loss curves saved for step {step}.")


if __name__ == "__main__":
    train()

