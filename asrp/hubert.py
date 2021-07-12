import joblib
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import soundfile as sf


class HubertCode(object):
    def __init__(self, hubert_model, km_path, km_layer):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model)
        self.model = HubertModel.from_pretrained(hubert_model)
        self.km_model = joblib.load(km_path)
        self.km_layer = km_layer
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()
            self.model = self.model.cuda()

    def __call__(self, filepath, sampling_rate=None):
        speech, sr = sf.read(filepath)
        input_values = self.processor(speech, return_tensors="pt", sampling_rate=sr).input_values
        if torch.cuda.is_available():
            input_values = input_values.cuda()
        hidden_states = self.model(input_values, output_hidden_states=True).hidden_states
        x = hidden_states[self.km_layer].squeeze()
        dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
        )
        return dist.argmin(dim=1).cpu().numpy()
