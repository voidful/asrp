from itertools import groupby
import gc

import joblib
import numpy
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from collections import defaultdict
from tqdm.contrib.concurrent import thread_map


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
            self.max_batch = self.get_max_batch()

    def get_max_batch(self):
        batch = 1
        try:
            while True:
                self.model(torch.rand([batch, self.chunk_length]).cuda())
                batch += 1
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        print("maximum batch size will be", batch)
        return batch

    def _process_feature(self, k, top_k=5, beamsearch=True, beamsize=3):
        feature = torch.cat(k, dim=0)
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
            'distance': dist.detach().cpu().numpy(),
            'center_diff': (feature.cpu() - torch.index_select(torch.tensor(self.C_np.transpose()).cpu(), 0,
                                                               min_dist.indices[:, 0].cpu())).numpy(),
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
            return_dict['beam_merged_code'] = [k for k, _ in groupby(code_output)]
        return return_dict

    def __call__(self, filepath='', input_values=[], beamsearch=True, top_k=3, beamsize=3):
        with torch.no_grad():
            if len(input_values) <= 0:
                speech, sr = torchaudio.load(filepath)
                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                    speech = resampler.forward(speech.squeeze(0)).numpy()
                else:
                    speech = speech.squeeze(0).numpy()
                input_values = self.processor(speech, return_tensors="pt",
                                              sampling_rate=self.sampling_rate).input_values

            if torch.cuda.is_available():
                input_values = input_values.cuda()

            chunks = list(torch.split(input_values, self.chunk_length, dim=1))
            if chunks[-1].shape[-1] < self.sampling_rate:
                concat_index = -2 if len(chunks) >= 2 else 0
                chunks[concat_index] = torch.cat(chunks[-2:], dim=-1)
                chunks = chunks[:concat_index + 1]

            batch_data = []
            cache_batch = []
            for audio_part in range(input_values.shape[0]):
                for c in chunks:
                    if len(cache_batch) >= self.max_batch:
                        batch_data.append(cache_batch)
                        cache_batch = []
                    cache_batch.append((c[audio_part], audio_part))
            if len(cache_batch) > 0:
                batch_data.append(cache_batch)

            features_dict = defaultdict(list)
            for bd in batch_data:
                for audio_id, hidden in zip([i[1] for i in bd],
                                            self.model(torch.stack([i[0] for i in bd], 0).to('cuda'),
                                                       output_hidden_states=True).hidden_states[self.km_layer]):
                    features_dict[audio_id].append(hidden)

            # Explicit clean up
            gc.collect()
            torch.cuda.empty_cache()
            return_dicts = thread_map(self._process_feature, features_dict.values(), max_workers=8)

            if len(return_dicts) == 1:
                return return_dicts[0]
            else:
                return return_dicts
