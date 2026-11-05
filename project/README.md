# FastSpeech 2: English and Telugu Text-to-Speech Synthesis

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, non-autoregressive Text-to-Speech (TTS) synthesis system implementing Microsoft's **FastSpeech 2** architecture independently for **English** and **Telugu**, trained on Indian acoustic datasets, with empirical evaluation of the Variance Adaptor across phonological and linguistic edge cases.

---

## 1. Project Directory Structure

The repository strictly follows the layout prescribed in the assignment guidelines:

```text
project/
├── English/
│   ├── train.py                 # English training loop with validation & loss logging
│   ├── preprocess.py            # Svarah dataset download, 5h/1h subsampling & feature extraction
│   ├── inference.py             # Synthesis of test sentences E1-E4 with variance adaptor plotting
│   ├── config.yaml              # Complete English hyperparameters & path definitions
│   └── checkpoints/             # Trained model checkpoints
├── Telugu/
│   ├── train.py                 # Telugu training loop with validation & loss logging
│   ├── preprocess.py            # IndicTTS dataset download, 5h/1h subsampling & feature extraction
│   ├── inference.py             # Synthesis of test sentences T1-T4 with variance adaptor plotting
│   ├── config.yaml              # Complete Telugu hyperparameters & path definitions
│   └── checkpoints/             # Trained model checkpoints
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/       # Synthesized WAVs: E1.wav, E2a.wav, E2b.wav, E3.wav, E4.wav
│   │   └── variance_adaptor_plots/
│   │       ├── duration/                # Predicted duration bar plots (frames per phoneme)
│   │       ├── pitch/                   # Continuous F0 pitch contour plots
│   │       └── energy/                  # Frame-level L2 energy envelope plots
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/       # Synthesized WAVs: T1.wav, T2a.wav, T2b.wav, T3.wav, T4.wav
│       └── variance_adaptor_plots/
│           ├── duration/                # Predicted duration bar plots (frames per phoneme)
│           ├── pitch/                   # Continuous F0 pitch contour plots
│           └── energy/                  # Frame-level L2 energy envelope plots
├── Report/
│   └── Technical_Report.pdf     # Publication-grade technical report
├── README.md                    # Comprehensive documentation and reproduction guide
└── requirements.txt             # Project dependencies
```

---

## 2. Model Architecture & Adaptations

Referring to *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (Ren et al., ICLR 2021) and the reference implementation (`ming024/FastSpeech2`), the architecture consists of:
1. **Feed-Forward Transformer (FFT) Encoder:** Multi-head self-attention with 1D-convolutional filter blocks.
2. **Variance Adaptor:**
   - **Duration Predictor & Length Regulator:** 2-layer 1D CNN with LayerNorm, ReLU, and Dropout predicting frame durations in log domain, followed by length regulation expansion.
   - **Pitch Predictor ($F_0$):** Continuous pitch predicted and quantized into 256 log-scale bins, projected through a pitch embedding table.
   - **Energy Predictor:** Frame-level STFT $L_2$-norm predicted and quantized into 256 bins, projected through an energy embedding table.
3. **FFT Decoder & Mel Linear Projection:** Converts expanded representations into 80-bin mel-spectrograms.
4. **HiFi-GAN Universal Neural Vocoder:** Inverts predicted mel-spectrograms into high-fidelity 22.05 kHz audio waveforms.

### Key Modifications Made Over the Reference GitHub Implementation:

| Component | Reference Implementation (`ming024`) | Our English Adaptation (`project/English/`) | Our Telugu Adaptation (`project/Telugu/`) |
| :--- | :--- | :--- | :--- |
| **Dataset Loader** | LJSpeech CSV single speaker | HuggingFace Parquet stream for `ai4bharat/Svarah` | HuggingFace Parquet stream for `SPRINGLab/IndicTTS_Telugu` |
| **G2P Engine** | CMUDict standard ARPAbet (fails on non-US / Indic words) | ARPAbet + rule-based decomposition for OOV and Indic loanwords (`andaroo`) | **Custom Indic Akshara G2P Engine**: Full Unicode range (`U+0C00`-`U+0C7F`), virama (`్`), conjuncts, and anusvara (`ం`) |
| **Text Normalizer** | US numbers only | Cardinals, ordinals, time (`10:45 AM`), dates (`12 November 2026`), typo-tolerant (`rhyth m`) | **Recursive Telugu Numeral Engine**: Converts any digit into spoken Telugu (`రెండు వేల ఇరవై ఆరు`, `పన్నెండవ`) |
| **Device Execution** | CUDA / CPU only (crashes on Apple MPS) | Fully device-agnostic (`torch.device` auto-detects CUDA, MPS, and CPU) | Fully device-agnostic (`torch.device` auto-detects CUDA, MPS, and CPU) |
| **Config Layout** | 3 separate yaml files | Single unified `config.yaml` as per assignment specification | Single unified `config.yaml` as per assignment specification |

---

## 3. Datasets & Subsampling Strategy

In strict compliance with assignment requirements, the training and validation splits were extracted with **zero overlap**:

| Language | Dataset | Total Available | Training Split | Validation Split | Overlap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | `ai4bharat/Svarah` | ~9.60 hours | **5.02 hours** (3,480 utterances) | **1.01 hours** (712 utterances) | **0.0% (Disjoint)** |
| **Telugu** | `SPRINGLab/IndicTTS_Telugu` | ~8.74 hours | **5.01 hours** (2,840 utterances) | **1.02 hours** (580 utterances) | **0.0% (Disjoint)** |

### Preprocessing Steps:
1. Audio resampled to 22,050 Hz and normalized to -27 dBFS.
2. Mel-Spectrogram: 80 bands, 1024 filter length, 256 hop length, 1024 window length.
3. Pitch ($F_0$): Extracted via PyWorld (Dio/Harvest), linearly interpolated across unvoiced regions, and log-scale normalized.
4. Energy: Calculated as $L_2$-norm of STFT frames ($\sqrt{\sum |S_k|^2}$).
5. Alignments: Extracted using phoneme-to-frame duration alignment.

---

## 4. Benchmark Test Sentences & Evaluation

All test sentences were synthesized **preserving the exact text as provided**:

### English Test Sentences:
- **E1:** *"The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."*
  - **Linguistic Focus:** 28-letter polysyllabic word (12 syllables). Evaluates stability of the Length Regulator without slurring or collapsing syllables.
- **E2a (Noun):** *"The orchestra's conduct during the ceremony was praised by the audience."*
  - **Linguistic Focus:** Heteronym noun /ˈkɒndʌkt/. Primary stress and higher F0 peak on initial syllable.
- **E2b (Verb):** *"The researchers will conduct an experiment to evaluate the proposed method."*
  - **Linguistic Focus:** Heteronym verb /kənˈdʌkt/. Stress and peak pitch shifted to second syllable with vowel reduction on the first.
- **E3:** *"The rhyth m of the orchestra changed unexpectedly."*
  - **Linguistic Focus:** Deliberate space error (`rhyth m`). Robustness to split token boundaries without acoustic failure.
- **E4:** *"The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."*
  - **Linguistic Focus:** Number/date/time expansion + code-mixed Telugu loanword (*"andaroo"* = everyone) in English.

### Telugu Test Sentences:
- **T1:** *"న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."*
  - **Linguistic Focus:** Medical compound loanword transliteration with complex conjuncts (`న్య`, `ర్ఫ`, `ర్మ`, `శ్లే`).
- **T2a:** *"తాడుపై అతని పట్టు చాలా బలంగా ఉంది."*
  - **Linguistic Focus:** Polysemy: "పట్టు" as physical grip/hold. Retroflex geminate closure duration and tight adverbial attachment.
- **T2b:** *"ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."*
  - **Linguistic Focus:** Polysemy: "పట్టు" as silk saree modifier forming compound noun phrase intonation.
- **T3:** *"పులి అడవిలో నిశ్శబ్దంగా నడిచింది."*
  - **Linguistic Focus:** Literary narrative cadence, geminates (`బ్ద`), and soft nasal transitions.
- **T4:** *"ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."*
  - **Linguistic Focus:** Formal administrative Telugu with dates, ordinals (`12వ`), years (`2026`), and time (`10:45`).

---

## 5. Quickstart & Reproducibility Guide

### Step 1: Environment Setup
```bash
git clone https://github.com/siddharthrajak49-commits/FastSpeech-2-Text-to-Speech-TTS-.git
cd FastSpeech-2-Text-to-Speech-TTS-/project

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Preprocess Datasets
Ensure your Hugging Face access token is available:
```bash
export HF_TOKEN="your_huggingface_access_token"

# Preprocess English (Svarah)
cd English
python preprocess.py
cd ..

# Preprocess Telugu (IndicTTS Telugu)
cd Telugu
python preprocess.py
cd ..
```

### Step 3: Train Models
```bash
# Train English FastSpeech 2
cd English
python train.py
cd ..

# Train Telugu FastSpeech 2
cd Telugu
python train.py
cd ..
```

### Step 4: Run Inference & Generate Plots
```bash
# Synthesize English sentences & generate variance plots
cd English
python inference.py
cd ..

# Synthesize Telugu sentences & generate variance plots
cd Telugu
python inference.py
cd ..
```

All generated audios will be located in `Results/{English,Telugu}/test_generated_audios/`, and variance adaptor plots will be in `Results/{English,Telugu}/variance_adaptor_plots/`.

---

## 6. Deliverables & Technical Report

The comprehensive technical report is available at:
📄 **[Report/Technical_Report.pdf](Report/Technical_Report.pdf)**

It includes:
- Detailed architectural breakdown and mathematical formulations
- Empirical analysis of training and validation loss convergence curves
- Spectrogram and variance adaptor plot interpretation for all 10 test sentences
- Comparative cross-lingual insights between stress-timed English and syllable-timed Telugu TTS
- Acoustic limitations, edge cases, and future directions
