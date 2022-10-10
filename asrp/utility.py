import random

import torch
import torchaudio


def cal_audio_time(filename):
    import subprocess, json
    result = subprocess.check_output(
        f'ffprobe -v quiet -show_format -of json "{filename}"',
        shell=True).decode()
    duration = json.loads(result)['format']['duration']
    return float(duration)


def shuffle_gen(n):
    # this is used like a range(n) list, but we don’t store
    # those entries where state[i] = i.
    state = dict()
    for remaining in range(n, 0, -1):
        i = random.randrange(remaining)
        yield state.get(i, i)
        state[i] = state.get(remaining - 1, remaining - 1)
        # Cleanup – we don’t need this information anymore
        state.pop(remaining - 1, None)


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def collate_fn_padd(batch, device):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in batch]).to(device)
    ## padd
    batch = [torch.Tensor(t).to(device) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=-100)
    ## compute mask
    mask = (batch != -100).to(device)
    return batch, lengths, mask
