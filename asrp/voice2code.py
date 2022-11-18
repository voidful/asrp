import gc
from collections import defaultdict
from functools import partial
from itertools import groupby

import joblib
import numpy
import torch
import torchaudio
from torch import nn
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
from transformers import Wav2Vec2FeatureExtractor, HubertModel


def collate_fn_pad(batch, device):
    '''
    Padds batch of variable length
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in batch]).to(device)
    ## padd
    batch = [torch.Tensor(t).to(device) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    ## compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class HubertCode(object):
    def __init__(self, hubert_model, km_path, km_layer, sampling_rate=16000, chunk_sec=10, worker=8, return_diff=False):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model)
        self.model = HubertModel.from_pretrained(hubert_model)
        self.model.eval()
        self.sampling_rate = sampling_rate
        self.chunk_length = sampling_rate * chunk_sec
        self.km_model = joblib.load(km_path)
        self.km_layer = km_layer
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.worker = worker
        self.return_diff = return_diff
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()
            self.model = self.model.cuda()
        self.max_batch = self.get_max_batch()

    def get_max_batch(self):
        print("calculating max batch size...")
        batch = 1
        with torch.no_grad():
            try:
                while True:
                    self.model(torch.rand([batch, self.chunk_length]).cuda())
                    batch += 2
                    gc.collect()
                    torch.cuda.empty_cache()
            except:
                pass
        batch = max(int(batch * 0.95), 1)
        print("maximum batch size will be", batch)
        return batch

    def _process_feature(self, k, top_k=5, feat_norm=True, beamsearch=False, beamsize=5):
        feature = torch.cat(k, dim=0) if isinstance(k, list) else k
        if feat_norm:
            m = nn.InstanceNorm1d(feature.shape[-1], affine=False)
            feature = m(feature)
        dist = torch.sqrt(
            feature.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(feature, self.C)
            + self.Cnorm
        )
        min_dist = torch.topk(dist.detach().cpu(), top_k, dim=-1, largest=False)
        pred_ind_array = min_dist.indices.cpu().numpy()
        pred_values_array = min_dist.values.cpu().numpy()
        code_output = min_dist.indices.T.cpu().numpy()[0]

        return_dict = {
            'code': list(code_output),
            'merged_code': [k for k, _ in groupby(code_output)]
        }
        if self.return_diff:
            return_dict.update({
                'distance': list(dist.detach().cpu().numpy()),
                'center_diff': list((feature.cpu() - torch.index_select(torch.tensor(self.C_np.transpose()).cpu(), 0,
                                                                        min_dist.indices[:, 0].cpu())).numpy()),
            })
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

    def list_prediction(self, filepaths, feat_norm=False, beamsearch=False, top_k=5, beamsize=5):
        with torch.no_grad():
            from torch.utils.data import Dataset
            from torch.utils.data import DataLoader

            class SpeechDataset(Dataset):
                def __init__(self, paths, processor, sampling_rate=16000):
                    self.paths = paths
                    self.processor = processor
                    self.sampling_rate = sampling_rate

                def __getitem__(self, index):
                    speech, sr = torchaudio.load(self.paths[index])
                    speech = speech.mean(0)
                    if sr != self.sampling_rate:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                        speech = resampler.forward(speech.squeeze(0))
                    else:
                        speech = speech.squeeze(0)
                    input_values = self.processor(speech, return_tensors="pt",
                                                  sampling_rate=self.sampling_rate).input_values
                    return input_values.squeeze(0)

                def __len__(self):
                    return len(self.paths)

            def dataloader_collate(batch):
                return torch.cat(batch, dim=0), [b.shape[0] for b in batch]

            dataset = SpeechDataset(filepaths, self.processor, self.sampling_rate)
            dataloader = DataLoader(dataset=dataset, batch_size=self.max_batch,
                                    shuffle=False,
                                    num_workers=self.worker,
                                    collate_fn=dataloader_collate)

            return_list = []
            for data_batch, size in tqdm(dataloader):
                batch_data = []
                batch_map_audio = []
                for b_id, audio in enumerate(torch.split(data_batch, size)):
                    for _ in torch.split(audio, self.chunk_length, dim=-1):
                        batch_data.append(_)
                        batch_map_audio.append(b_id)

                code_result = defaultdict(list)
                for bd, bm in zip(chunks(batch_data, self.max_batch), chunks(batch_map_audio, self.max_batch)):
                    batch, lengths, masks = collate_fn_pad(bd, self.device)
                    masks_ratio = lengths / torch.max(lengths)
                    hidden = self.model(batch,
                                        output_hidden_states=True).hidden_states[self.km_layer].detach()
                    mask_len = (hidden.shape[1] * masks_ratio).int()
                    for a, h, ml in zip(bm, hidden, mask_len):
                        code_result[a].append(h[:ml, :])

                for k, v in code_result.items():
                    result = {}
                    for d in thread_map(
                            partial(self._process_feature,
                                    top_k=top_k,
                                    beamsearch=beamsearch,
                                    beamsize=beamsize,
                                    feat_norm=feat_norm), v,
                            leave=False):
                        for k2, v2 in d.items():
                            if k2 in result:
                                result[k2].extend(v2)
                            else:
                                result[k2] = v2
                    return_list.append(result)
        return return_list

    def __call__(self, filepath='', input_values=[], feat_norm=False, beamsearch=False, top_k=5, beamsize=5):
        with torch.no_grad():
            if len(input_values) <= 0:
                speech, sr = torchaudio.load(filepath)
                speech = speech.mean(0).unsqueeze(0)
                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                    speech = resampler.forward(speech.squeeze(0)).numpy()
                else:
                    speech = speech.squeeze(0).numpy()
                input_values = self.processor(speech, return_tensors="pt",
                                              sampling_rate=self.sampling_rate).input_values

            code_result = []
            batch, lengths, masks = collate_fn_pad(input_values, self.device)
            masks_ratio = lengths / torch.max(lengths)
            hidden = self.model(batch,
                                output_hidden_states=True).hidden_states[self.km_layer].detach()
            mask_len = (hidden.shape[1] * masks_ratio).int()
            code_result.append(hidden[:mask_len, :].squeeze(0))

            result = {}
            for d in thread_map(partial(self._process_feature,
                                        top_k=top_k,
                                        beamsearch=beamsearch,
                                        beamsize=beamsize,
                                        feat_norm=feat_norm),
                                code_result, leave=False):
                for k2, v2 in d.items():
                    if k2 in result:
                        result[k2].extend(v2)
                    else:
                        result[k2] = v2
            return result
