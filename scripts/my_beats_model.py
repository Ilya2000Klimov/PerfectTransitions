import os
import torch
import torchaudio
import yaml
import numpy as np

from fairseq import checkpoint_utils, tasks
from fairseq.models import BaseFairseqModel
from fairseq.data.audio.audio_utils import get_features_or_waveform


def load_beats_model(config_path):
    """
    Loads a BEATs model via Fairseq, using the official config + checkpoint.
    Args:
        config_path (str): Path to the YAML config (e.g. 'BEATs_iter3_AS2M.yaml').
                           This config should point to the actual model checkpoint
                           under 'task.init_model'.
    Returns:
        model (torch.nn.Module): A ready-to-infer BEATs model in eval mode (CPU by default).
        layer_for_extraction (int): The layer index used for feature extraction (default=12).
                                    You can change if you want deeper or shallower embeddings.
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # The model checkpoint path is stored here:
    model_path = config['task']['init_model']
    # Optional: you might choose which layer to extract from:
    layer_for_extraction = config['task'].get('layer', 12)

    # Load model weights into CPU memory
    ckpt = checkpoint_utils.load_checkpoint_to_cpu(model_path)

    # Build the fairseq task from config
    task_cfg = config['task']
    task = tasks.setup_task(task_cfg)

    # Build the actual model
    model = BaseFairseqModel.build_model_from_checkpoint(ckpt, task)
    model.eval()  # inference mode by default

    # Some BEATs versions require model.w2v_model to be set to eval as well:
    if hasattr(model, 'w2v_model'):
        model.w2v_model.eval()

    return model, layer_for_extraction


def compute_beats_embeddings(
    audio_path,
    model_tuple,
    device,
    sr=16000,
    mono=True
):
    """
    Loads an audio file, resamples to 16kHz, passes it through the BEATs model,
    and returns (T, D) embeddings from a specified layer.

    Args:
        audio_path (str): Path to audio file (wav/flac/mp3).
        model_tuple: (model, layer_for_extraction) from load_beats_model().
        device: Torch device (cpu or cuda).
        sr (int): Target sample rate, default 16kHz for BEATs.
        mono (bool): Whether to convert to mono if multi-channel.

    Returns:
        np.ndarray: A numpy array of shape (T, D) with time-step embeddings.
    """
    model, layer_idx = model_tuple

    # 1) Load raw waveform with torchaudio
    waveform, source_sr = torchaudio.load(audio_path)

    # 2) Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 3) Resample if necessary
    if source_sr != sr:
        resampler = torchaudio.transforms.Resample(source_sr, sr)
        waveform = resampler(waveform)

    # waveform shape is [1, num_samples]
    waveform = waveform.to(device)

    # 4) Forward pass through BEATs
    # According to the official example, we disable mask, use features_only=True
    with torch.no_grad():
        # Some versions: model(waveform, mask=False, features_only=True, layer=layer_idx)
        # Others: model.extract_features(waveform, ...). 
        # We'll try the "model(...)" approach:

        model_out = model(
            waveform, mask=False, features_only=True, layer=layer_idx
        )
        # Typically model_out is a dict with 'x': [B, T, D]
        # We only have batch_size=1, so shape is [1, T, D]
        feats = model_out['x']  # [1, T, D]

    feats = feats.squeeze(0)  # -> [T, D]
    feats_np = feats.cpu().numpy()

    return feats_np
