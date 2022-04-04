from itertools import groupby

import joblib
import numpy
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel


class HubertCode(object):
    def __init__(self, hubert_model, km_path, km_layer, sampling_rate=16000, chunk_sec=5):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model)
        self.model = HubertModel.from_pretrained(hubert_model)
        self.model.eval()
        self.sampling_rate = sampling_rate
        self.chunk_length = sampling_rate * chunk_sec
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

    def __call__(self, filepath, beamsearch=True, top_k=10, beamsize=200):
        with torch.no_grad():
            speech, sr = torchaudio.load(filepath)
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                speech = resampler.forward(speech.squeeze(0)).numpy()
            else:
                speech = speech.squeeze(0).numpy()
            input_values = self.processor(speech, return_tensors="pt", sampling_rate=self.sampling_rate).input_values
            if torch.cuda.is_available():
                input_values = input_values.cuda()

            chunks = torch.split(input_values, self.chunk_length, dim=1)
            for i, chunk in enumerate(chunks):
                hidden_states = self.model(chunk, output_hidden_states=True).hidden_states
                feat = hidden_states[self.km_layer].squeeze()
                if i == 0:
                    feature = feat
                else:
                    feature = torch.cat([feature, feat], dim=0)

            dist = torch.sqrt(
                feature.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(feature, self.C)
                + self.Cnorm
            )
            min_dist = torch.topk(dist.detach(), top_k, dim=-1, largest=False)
            pred_ind_array = min_dist.indices.cpu().numpy()
            pred_values_array = min_dist.values.cpu().numpy()
            code_output = min_dist.indices.T.cpu().numpy()[0]

            return_dict = {
                'code': code_output,
                'distance': dist.detach().cpu(),
                'center_diff': feature.cpu() - torch.index_select(torch.tensor(self.C_np.transpose()).cpu(), 0,
                                                                  min_dist.indices[:, 0].cpu()),
                'merged_code': [k for k, _ in groupby(code_output)]
            }
            if beamsearch:
                sequences = [[[], 1.0]]
                for i_row, v_row in zip(pred_ind_array, pred_values_array):
                    all_candidates = list()
                    for seq in sequences:
                        tokens, score = seq
                        for k, v in zip(i_row, v_row):
                            norm_len_rate = (len([k for k, _ in groupby(tokens + [k])]) / len(code_output))
                            norm_dist_rate = (v / numpy.sum(v_row))
                            candidate = [tokens + [k], score + norm_len_rate * norm_dist_rate]
                            all_candidates.append(candidate)
                    ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=False)
                    sequences = ordered[:beamsize]
                code_output = sequences[0][0]
                return_dict['beam_code'] = code_output
                return_dict['beam_merged_code'] = code_output

            return return_dict
