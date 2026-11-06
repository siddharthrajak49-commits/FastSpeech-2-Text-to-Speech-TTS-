import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


def safe_torch_load(path, map_location=None):
    """
    Universal checkpoint loader supporting PyTorch 1.x, 2.0-2.5, and 2.6+ (weights_only=True default).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = safe_torch_load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hifi_dir = os.path.join(base_dir, "hifigan")
    if not os.path.exists(hifi_dir):
        hifi_dir = "hifigan"

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        cfg_path = os.path.join(hifi_dir, "config.json")
        with open(cfg_path, "r") as f:
            voc_config = json.load(f)
        voc_config = hifigan.AttrDict(voc_config)
        vocoder = hifigan.Generator(voc_config)
        
        ckpt_filename = "generator_universal.pth.tar" if speaker == "universal" else "generator_LJSpeech.pth.tar"
        ckpt_path = os.path.join(hifi_dir, ckpt_filename)
        
        if os.path.exists(ckpt_path):
            ckpt = safe_torch_load(ckpt_path, map_location=device)
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)
        else:
            print(f"Warning: HiFi-GAN checkpoint not found at {ckpt_path}. Returning uninitialized.")

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
