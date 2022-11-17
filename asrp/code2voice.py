import collections
import sys
from random import random

from asrp import glow

from asrp.hifigan import CodeHiFiGANVocoder

sys.modules['glow'] = glow
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from asrp.tacotron2 import Tacotron2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict(
                (key, _apply(value)) for key, value in x.items()
            )
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


class Code2Speech(object):
    def __init__(self, tts_checkpoint, waveglow_checkpint=None, vocoder='tecotron', max_decoder_steps=2000,
                 end_tok=None, code_begin_pad=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocoder = vocoder
        if self.vocoder == 'tecotron':
            self.tacotron_model, self.sample_rate, self.hparams = load_tacotron(
                tacotron_model_path=tts_checkpoint,
                max_decoder_steps=max_decoder_steps,
            )
            self.waveglow, self.denoiser = load_waveglow(waveglow_path=waveglow_checkpint)
            self.tacotron_model = self.tacotron_model.to(self.device)
            self.waveglow = self.waveglow.to(self.device)
            self.denoiser = self.denoiser.to(self.device)
        elif self.vocoder == 'hifigan':
            self.sample_rate = 16000
            self.hifigan = load_hifigan(model_path=tts_checkpoint)

        self.end_tok = end_tok
        self.code_begin_pad = code_begin_pad

    def __call__(self, code, strength=0.1):
        with torch.no_grad():
            code = [i + self.code_begin_pad for i in code]
            if self.end_tok is not None and code[-1] != self.end_tok:
                code.append(self.end_tok)
            tts_input = torch.tensor(code)
            if self.vocoder == 'tecotron':
                tts_input.to(device)
                mel, aud, aud_dn, has_eos = synthesize_audio(
                    self.tacotron_model,
                    self.waveglow,
                    self.denoiser,
                    tts_input.unsqueeze(0),
                    strength=strength
                )
                audio_seq = aud_dn[0].cpu().float().numpy()
            elif self.vocoder == 'hifigan':
                x = {
                    "code": tts_input.view(1, -1),
                }
                # x["dur_prediction"] = dur_prediction
                # if self.hifigan.multispkr:
                #     spk = (
                #         random.randint(0, vocnum_speakers - 1)
                #         if args.speaker_id == -1
                #         else args.speaker_id
                #     )
                #     suffix = f"_spk{spk}"
                #     x["spkr"] = torch.LongTensor([spk]).view(1, 1)
                audio_seq = self.hifigan(x, False)

            return audio_seq


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis framesa
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    import librosa
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa.util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa.util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            import librosa
            assert (filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = librosa.util.pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            import librosa
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > librosa.util.tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, waveglow, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros'):
        super(Denoiser, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.stft = STFT(filter_length=filter_length,
                         hop_length=int(filter_length / n_overlap),
                         win_length=win_length).to(self.device)
        if mode == 'zeros':
            mel_input = torch.zeros(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype,
                device=waveglow.upsample.weight.device)
        elif mode == 'normal':
            mel_input = torch.randn(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype,
                device=waveglow.upsample.weight.device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = waveglow.infer(mel_input, sigma=0.0).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.to(self.device).float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised


def load_quantized_audio_from_file(file_path):
    base_fname_batch, quantized_units_batch = [], []
    with open(file_path) as f:
        for line in f:
            base_fname, quantized_units_str = line.rstrip().split("|")
            quantized_units = [int(q) for q in quantized_units_str.split(" ")]
            base_fname_batch.append(base_fname)
            quantized_units_batch.append(quantized_units)
    return base_fname_batch, quantized_units_batch


def synthesize_audio(model, waveglow, denoiser, inp, lab=None, strength=0.0):
    assert inp.size(0) == 1
    inp = inp.to(device)
    if lab is not None:
        lab = torch.LongTensor(1).to(device).fill_(lab)

    with torch.no_grad():
        _, mel, _, ali, has_eos = model.inference(inp, lab, ret_has_eos=True)
        aud = waveglow.infer(mel, sigma=0.666)
        aud_dn = denoiser(aud, strength=strength).squeeze(1)
    return mel, aud, aud_dn, has_eos


def load_tacotron(tacotron_model_path, max_decoder_steps):
    ckpt_dict = torch.load(tacotron_model_path, map_location=torch.device(device))
    hparams = ckpt_dict["hparams"]
    hparams.max_decoder_steps = max_decoder_steps
    sr = hparams.sampling_rate
    model = Tacotron2(hparams)
    model.load_state_dict(ckpt_dict["model_dict"])
    model = model.eval().half()
    return model, sr, hparams


def load_waveglow(waveglow_path):
    if waveglow_path is None:
        waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
    else:
        waveglow = torch.load(waveglow_path, map_location=torch.device(device))["model"]

    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.eval().half().to(device)
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    return waveglow, denoiser


def load_hifigan(model_path, speaker_id=0, num_speakers=200):
    vocoder = CodeHiFiGANVocoder(model_path)
    multispkr = vocoder.model.multispkr
    if multispkr:
        assert (
                speaker_id < num_speakers
        ), f"invalid --speaker-id ({speaker_id}) with total #speakers = {num_speakers}"
    vocoder.multispkr = multispkr
    vocoder.speaker_id = speaker_id
    vocoder.num_speakers = num_speakers
    return vocoder
