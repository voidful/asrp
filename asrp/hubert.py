from itertools import groupby

import joblib
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import soundfile as sf


class HubertCode(object):
    def __init__(self, hubert_model, km_path, km_layer, return_diff=False):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model)
        self.model = HubertModel.from_pretrained(hubert_model)
        self.model.eval()
        self.km_model = joblib.load(km_path)
        self.km_layer = km_layer
        self.return_diff = return_diff
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()
            self.model = self.model.cuda()

    def __call__(self, filepath, sampling_rate=None, merge=True):
        with torch.no_grad():
            speech, sr = sf.read(filepath)
            input_values = self.processor(speech, return_tensors="pt", sampling_rate=sr).input_values
            if torch.cuda.is_available():
                input_values = input_values.cuda()
            hidden_states = self.model(input_values, output_hidden_states=True).hidden_states
            x = hidden_states[self.km_layer].squeeze()
            dist = torch.sqrt(
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            min_dist = dist.detach().min(dim=1)
            if merge_result:
                unitcode = [k for k, _ in groupby(min_dist.indices.cpu().numpy())]
            else:
                unitcode = min_dist.indices.cpu().numpy()
            if self.return_diff:
                return unitcode, x.cpu() - torch.index_select(
                    torch.tensor(self.C_np.transpose()).cpu(), 0, min_dist.indices.cpu())
            else:
                return unitcode
