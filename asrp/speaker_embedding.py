import os
from pathlib import Path

import nlp2
import torch
import torchaudio


def extract_d_vector(wav_path=None, wav_tensor=None, sampling_rate=16000, inference=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.inference_mode(inference):
        try:
            save_folder = Path(__file__).parent
        except:
            save_folder = './'
        wav2mel_path = os.path.join(save_folder, 'wav2mel.pt')
        dvector_path = os.path.join(save_folder, 'dvector-step250000.pt')
        if not os.path.exists(wav2mel_path):
            nlp2.download_file(
                'https://github.com/yistLin/dvector/releases/download/v1.1.1/wav2mel.pt',
                f'{Path(__file__).parent}')
        if not os.path.exists(dvector_path):
            nlp2.download_file(
                'https://github.com/yistLin/dvector/releases/download/v1.1.1/dvector-step250000.pt',
                f'{Path(__file__).parent}')
        wav2mel = torch.jit.load(wav2mel_path)
        dvector = torch.jit.load(dvector_path).eval().to(device)
        if not torch.is_tensor(wav_tensor):
            wav_tensor, sample_rate = torchaudio.load(wav_path)
            mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
        else:
            mel_tensor = wav2mel(wav_tensor, sampling_rate)
        emb_tensor = dvector.embed_utterance(mel_tensor.to(device))  # shape: (emb_dim)
    return emb_tensor.tolist()


def extract_x_vector(wav_path=None, wav_tensor=None, sampling_rate=16000, inference=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.inference_mode(inference):
        from speechbrain.pretrained import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                    savedir="pretrained_models/spkrec-xvect-voxceleb",
                                                    run_opts={"device": device})
        if not torch.is_tensor(wav_tensor):
            wav_tensor, sampling_rate = torchaudio.load(wav_path)
        embeddings = classifier.encode_batch(wav_tensor.to(device)).squeeze()
        return embeddings.tolist()
