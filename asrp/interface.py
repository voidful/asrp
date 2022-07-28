import os.path

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor


class HFSpeechInference:
    def __init__(self, model_name, onnx_path='',
                 beam_width=100, hotwords=[], hotword_weight=20,
                 alpha=0.7, beta=1.5,
                 beam_prune_logp=-10.0,
                 token_min_logp=-10.0,
                 use_auth_token=False):
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
