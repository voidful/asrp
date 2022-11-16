import os.path
from typing import Optional, Union, List, Iterable

import numpy as np
import torch

try:
    import whisper
except:
    pass
from math import exp
from transformers import AutoModelForCTC, AutoProcessor


class HFSpeechInference:
    def __init__(self, model_name, onnx_path='',
                 beam_width=100, hotwords=[], hotword_weight=20,
                 alpha=0.7, beta=1.5,
                 beam_prune_logp=-10.0,
                 token_min_logp=-10.0,
                 use_auth_token=False,
                 homophone_extend=False):
        self.processor = AutoProcessor.from_pretrained(model_name, use_auth_token=use_auth_token)
        self.model = AutoModelForCTC.from_pretrained(model_name, use_auth_token=use_auth_token)
        self.beam_width = beam_width
        self.hotwords = hotwords
        self.hotword_weight = hotword_weight
        self.alpha = alpha
        self.beam_prune_logp = beam_prune_logp
        self.token_min_logp = token_min_logp
        self.beta = beta
        self.is_onnx = False
        if homophone_extend and hasattr(self.processor, 'decoder'):
            self.processor.decoder.enable_homophone_extend = homophone_extend
            print("enable homophone extend")

        if os.path.exists(onnx_path):
            import onnxruntime as rt
            options = rt.SessionOptions()
            options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.model = rt.InferenceSession(onnx_path, options)
            self.is_onnx = True

    def buffer_to_text(self, audio_buffer):
        if (len(audio_buffer) == 0):
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            if self.is_onnx:
                logits = \
                    self.model.run(None, {self.model.get_inputs()[0].name: inputs.input_values.cpu().numpy()})[0]
            else:
                logits = self.model(inputs.input_values).logits.cpu().numpy()

        if hasattr(self.processor, 'decoder'):
            transcription = \
                self.processor.decode(logits[0],
                                      beam_width=self.beam_width,
                                      hotwords=self.hotwords,
                                      hotword_weight=self.hotword_weight,
                                      alpha=self.alpha, beta=self.beta,
                                      token_min_logp=self.token_min_logp,
                                      beam_prune_logp=self.beam_prune_logp,
                                      output_word_offsets=True,
                                      lm_score_boundary=True)
            transcription = transcription.text
        else:
            predicted_ids = np.argmax(logits, axis=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.lower()

    def file_to_text(self, filename):
        import librosa
        audio_input, samplerate = librosa.load(filename, sr=16000)
        return self.buffer_to_text(audio_input)


class WhisperInference:
    def __init__(self, model_size='base',
                 task: str = "transcribe",
                 speech_threshold: float = 0.4,
                 temperature: float = 0.0,
                 sample_len: Optional[int] = None,  # maximum number of tokens to sample
                 best_of: Optional[int] = None,  # number of independent samples to collect, when t > 0
                 beam_size: Optional[int] = None,  # number of beams in beam search, when t == 0
                 patience: Optional[float] = None,  # patience in beam search (https://arxiv.org/abs/2204.05424)
                 length_penalty: Optional[float] = None,  # "alpha" in Google NMT, None defaults to length norm
                 prompt: Optional[Union[str, List[int]]] = None,  # text or tokens for the previous context
                 prefix: Optional[Union[str, List[int]]] = None,  # text or tokens to prefix the current context
                 suppress_blank: bool = True,  # this will suppress blank outputs
                 suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1",
                 without_timestamps: bool = False,  # use <|notimestamps|> to sample text tokens only
                 max_initial_timestamp: Optional[float] = 1.0,  # the initial timestamp cannot be later than this
                 language=None):
        self.model = whisper.load_model(model_size)
        self.speech_threshold = speech_threshold
        self.options = whisper.DecodingOptions(fp16=False,
                                               task=task,
                                               language=language,
                                               temperature=temperature,
                                               sample_len=sample_len,
                                               best_of=best_of,
                                               beam_size=beam_size,
                                               patience=patience,
                                               length_penalty=length_penalty,
                                               prompt=prompt,
                                               prefix=prefix,
                                               suppress_blank=suppress_blank,
                                               suppress_tokens=suppress_tokens,
                                               without_timestamps=without_timestamps,
                                               max_initial_timestamp=max_initial_timestamp)

    def buffer_to_text(self, audio_buffer):
        if (len(audio_buffer) == 0):
            return ""

        with torch.no_grad():
            audio = whisper.pad_or_trim(audio_buffer)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            result = whisper.decode(self.model, mel, self.options)
        if exp(result.avg_logprob) > self.speech_threshold:
            return result.text.lower()
        else:
            return ""

    def file_to_text(self, filename):
        import librosa
        audio_input, samplerate = librosa.load(filename, sr=16000)
        return self.buffer_to_text(audio_input)
