from pathlib import Path

import torch
import torchaudio

from .denoiser.pretrained import master64
from .denoiser.utils import cal_snr
from .utility import write, collate_fn_padd


class SpeechEnhancer:
    def __init__(self, device='cpu', dry_wet=0.01, sampling_rate=16000, chunk_sec=30, max_batch=3):
        torchaudio.set_audio_backend("sox_io")  # switch backend
        self.device = device
        self.dry_wet = dry_wet
        self.sampling_rate = sampling_rate
        self.chunk_sec = chunk_sec
        self.max_batch = max_batch
        self.chunk_length = sampling_rate * chunk_sec

        self.enhance_model = master64()
        self.enhance_model = self.enhance_model.to(self.device)
        self.enhance_model.eval()

    def __call__(self,
                 filepath='',
                 input_values=[],
                 result_path=''):

        if len(filepath) > 0:
            out, sr = torchaudio.load(filepath)
            out = out.mean(0).unsqueeze(0)
        else:
            out = input_values

        # split audio into chunks
        chunks = list(torch.split(out, self.chunk_length, dim=1))
        if chunks[-1].shape[-1] < self.sampling_rate:
            concat_index = -2 if len(chunks) >= 2 else 0
            chunks[concat_index] = torch.cat(chunks[-2:], dim=-1)
            chunks = chunks[:concat_index + 1]

        batch_data = []
        cache_batch = []
        for c in chunks:
            if len(cache_batch) >= self.max_batch:
                batch_data.append(cache_batch)
                cache_batch = []
            cache_batch.append(c)
        if len(cache_batch) > 0:
            batch_data.append(cache_batch)

        enhance_result = []
        for bd in batch_data:
            batch, lengths, masks = collate_fn_padd([i[0] for i in bd], self.device)
            estimate = (1 - self.dry_wet) * self.enhance_model(batch).squeeze(1) + self.dry_wet * batch
            enhance_result.append(torch.masked_select(estimate, masks).detach().cpu())

        denoise = torch.cat(enhance_result, dim=-1).unsqueeze(0)

        p = Path(filepath)
        write(denoise, str(Path(p.parent, f"{p.stem}_enhanced{p.suffix}")), sr)
        snr = cal_snr(out, denoise)
        snr = snr.cpu().detach().numpy()[0]
        return snr
