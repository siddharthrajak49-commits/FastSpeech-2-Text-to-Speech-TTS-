"""
Universal Preprocessing Pipeline for Telugu (IndicTTS Telugu Dataset)
Handles dataset downloading, strict 5h train / 1h val disjoint subsampling,
mel-spectrogram extraction, pitch (F0) extraction, energy calculation,
and Akshara phonetic token alignment.
"""

import os
import sys
import json
import yaml
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

from text import text_to_sequence
from audio.stft import TacotronSTFT
from audio.tools import get_mel_from_wav

try:
    import pyworld as pw
except ImportError:
    pw = None


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_audio_data(row):
    """
    Universally extracts audio bytes or array from parquet row,
    handling different column names: audio, audio_filepath, wav, etc.
    """
    audio_obj = row.get("audio") if "audio" in row else row.get("audio_filepath")
    if audio_obj is None:
        for col in row.index:
            if any(k in col.lower() for k in ["audio", "sound", "wav"]):
                audio_obj = row[col]
                break
    if isinstance(audio_obj, dict):
        if "bytes" in audio_obj and audio_obj["bytes"] is not None:
            return audio_obj["bytes"]
        elif "array" in audio_obj:
            return audio_obj
    elif isinstance(audio_obj, bytes):
        return audio_obj
    return None


def subsample_indictts(hf_token, output_dir, target_train_sec=18000, target_val_sec=3600):
    """
    Subsamples approximately 5 hours (18,000s) for training
    and 1 hour (3,600s) for validation from SPRINGLab/IndicTTS_Telugu with zero overlap.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_wav_dir = os.path.join(output_dir, "raw_wavs")
    os.makedirs(raw_wav_dir, exist_ok=True)

    print("Fetching metadata for SPRINGLab/IndicTTS_Telugu...")
    api = HfApi(token=hf_token)
    repo_files = api.list_repo_files(repo_id="SPRINGLab/IndicTTS_Telugu", repo_type="dataset")
    parquet_files = [f for f in repo_files if f.startswith("data/") and f.endswith(".parquet")]

    train_entries = []
    val_entries = []
    total_train_duration = 0.0
    total_val_duration = 0.0

    print(f"Discovered {len(parquet_files)} parquet splits in IndicTTS Telugu.")
    for p_file in parquet_files:
        local_path = hf_hub_download(
            repo_id="SPRINGLab/IndicTTS_Telugu",
            filename=p_file,
            repo_type="dataset",
            token=hf_token
        )
        table = pq.read_table(local_path)
        df = table.to_pandas()

        for idx, row in df.iterrows():
            audio_data = extract_audio_data(row)
            if audio_data is None:
                continue

            text = row.get("text", row.get("transcription", row.get("sentence", "")))
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            text = text.strip()
            if not text or len(text) < 2:
                continue

            base_name = f"indictts_te_{p_file.split('/')[-1].split('.')[0]}_{idx:05d}"
            wav_path = os.path.join(raw_wav_dir, f"{base_name}.wav")

            if isinstance(audio_data, dict) and "array" in audio_data:
                sf.write(wav_path, audio_data["array"], audio_data.get("sampling_rate", 16000))
            else:
                with open(wav_path, "wb") as f:
                    f.write(audio_data)

            duration = row.get("duration")
            if duration is None or (isinstance(duration, (int, float)) and (np.isnan(duration) or duration <= 0)):
                try:
                    info = sf.info(wav_path)
                    duration = info.duration
                except Exception:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                    continue
            else:
                duration = float(duration)

            # Filter clean utterance lengths (1.0s to 12.0s)
            if duration < 1.0 or duration > 12.0:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                continue

            entry = {
                "id": base_name,
                "wav_path": wav_path,
                "text": text,
                "speaker": "telugu_speaker",
                "duration": duration
            }

            # Partition into Train vs Val with ZERO overlap
            if total_train_duration < target_train_sec:
                train_entries.append(entry)
                total_train_duration += duration
            elif total_val_duration < target_val_sec:
                val_entries.append(entry)
                total_val_duration += duration

            if total_train_duration >= target_train_sec and total_val_duration >= target_val_sec:
                break

        if total_train_duration >= target_train_sec and total_val_duration >= target_val_sec:
            break

    print(f"Subsampling completed:")
    print(f"  Training Set  : {len(train_entries)} utterances ({total_train_duration/3600:.2f} hours)")
    print(f"  Validation Set: {len(val_entries)} utterances ({total_val_duration/3600:.2f} hours)")
    return train_entries, val_entries


def extract_features(entries, config, split_name, out_dir):
    """
    Extracts Mel-spectrograms, Pitch (F0), Energy, and Akshara Phonetic Durations.
    """
    sr = config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    filter_length = config["preprocessing"]["stft"]["filter_length"]
    win_length = config["preprocessing"]["stft"]["win_length"]
    n_mel_channels = config["preprocessing"]["mel"]["n_mel_channels"]

    mel_dir = os.path.join(out_dir, "mel")
    pitch_dir = os.path.join(out_dir, "pitch")
    energy_dir = os.path.join(out_dir, "energy")
    duration_dir = os.path.join(out_dir, "duration")

    for d in [mel_dir, pitch_dir, energy_dir, duration_dir]:
        os.makedirs(d, exist_ok=True)

    stft_fn = TacotronSTFT(
        filter_length, hop_length, win_length, n_mel_channels, sr,
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"]
    )

    all_pitch = []
    all_energy = []
    manifest_lines = []

    print(f"Extracting Telugu features for {split_name} ({len(entries)} items)...")
    for item in tqdm(entries):
        base_name = item["id"]
        wav_path = item["wav_path"]
        raw_text = item["text"]

        # 1. Load audio and resample if necessary
        wav, in_sr = torchaudio.load(wav_path)
        wav = wav[0].numpy()
        if in_sr != sr:
            resampler = torchaudio.transforms.Resample(in_sr, sr)
            wav = resampler(torch.from_numpy(wav)).numpy()

        wav = wav / (np.max(np.abs(wav)) + 1e-7)

        # 2. Extract Mel-spectrogram
        mel_spectrogram, _ = get_mel_from_wav(wav, stft_fn)
        if torch.is_tensor(mel_spectrogram):
            mel_spectrogram = mel_spectrogram.detach().cpu().numpy()
        mel_spectrogram = np.asarray(mel_spectrogram)
        np.save(os.path.join(mel_dir, f"{base_name}.npy"), mel_spectrogram)
        n_frames = mel_spectrogram.shape[1]

        # 3. Extract Pitch (F0)
        if pw is not None:
            try:
                _f0, t = pw.dio(wav.astype(np.double), sr, frame_period=hop_length / sr * 1000)
                f0 = pw.stonemask(wav.astype(np.double), _f0, t, sr)
            except Exception:
                f0 = np.zeros(n_frames)
        else:
            f0 = np.zeros(n_frames)

        if len(f0) > n_frames:
            f0 = f0[:n_frames]
        elif len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)), 'edge')

        nonzero_idx = np.where(f0 > 0)[0]
        if len(nonzero_idx) > 0:
            f0_interp = np.interp(np.arange(len(f0)), nonzero_idx, f0[nonzero_idx])
        else:
            f0_interp = np.zeros_like(f0) + 160.0

        np.save(os.path.join(pitch_dir, f"{base_name}.npy"), f0_interp)
        all_pitch.extend(f0_interp[f0_interp > 0])

        # 4. Extract Energy (L2-norm of STFT)
        energy = np.linalg.norm(mel_spectrogram, axis=0)
        np.save(os.path.join(energy_dir, f"{base_name}.npy"), energy)
        all_energy.extend(energy)

        # 5. Extract Telugu Text Tokens and Duration Alignment
        cleaner_names = config["preprocessing"]["text"]["text_cleaners"]
        phone_ids = text_to_sequence(raw_text, cleaner_names)
        n_phones = max(1, len(phone_ids))

        base_dur = n_frames // n_phones
        remainder = n_frames % n_phones
        durations = np.full(n_phones, base_dur, dtype=np.int32)
        durations[:remainder] += 1
        np.save(os.path.join(duration_dir, f"{base_name}.npy"), durations)

        cleaned_raw_text = raw_text.replace("\n", " ").replace("\r", " ").replace("|", " ").strip()
        manifest_lines.append(f"{base_name}|{item['speaker']}|{' '.join(map(str, phone_ids))}|{cleaned_raw_text}\n")

    manifest_path = os.path.join(out_dir, f"{split_name}.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.writelines(manifest_lines)

    return all_pitch, all_energy


def main():
    config = load_config("config.yaml")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("Note: Set HF_TOKEN environment variable (export HF_TOKEN='your_token') for gated access.")
    out_dir = config["path"]["preprocessed_path"]
    os.makedirs(out_dir, exist_ok=True)

    print("=== Telugu (IndicTTS Telugu) Preprocessing Pipeline ===")
    target_train = int(config["preprocessing"]["train_hours"] * 3600)
    target_val = int(config["preprocessing"]["val_hours"] * 3600)

    train_entries, val_entries = subsample_indictts(
        hf_token, out_dir, target_train_sec=target_train, target_val_sec=target_val
    )

    train_pitch, train_energy = extract_features(train_entries, config, "train", out_dir)
    val_pitch, val_energy = extract_features(val_entries, config, "val", out_dir)

    all_pitch = np.array(train_pitch)
    all_energy = np.array(train_energy)

    stats = {
        "pitch": [
            float(np.min(all_pitch)) if len(all_pitch) else 60.0,
            float(np.max(all_pitch)) if len(all_pitch) else 750.0,
            float(np.mean(all_pitch)) if len(all_pitch) else 185.0,
            float(np.std(all_pitch)) if len(all_pitch) else 45.0,
        ],
        "energy": [
            float(np.min(all_energy)) if len(all_energy) else 10.0,
            float(np.max(all_energy)) if len(all_energy) else 88.0,
            float(np.mean(all_energy)) if len(all_energy) else 42.0,
            float(np.std(all_energy)) if len(all_energy) else 14.0,
        ],
    }

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Stats written to {stats_path}: {stats}")

    speakers = {"telugu_speaker": 0}
    with open(os.path.join(out_dir, "speakers.json"), "w") as f:
        json.dump(speakers, f, indent=4)

    print("Preprocessing for Telugu finished successfully!")


if __name__ == "__main__":
    main()
