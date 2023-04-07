import nlp2

from asrp import HubertCode


def mhubert_layer11_code1000(sampling_rate=16000, chunk_sec=10, worker=8,
                             return_diff=False,
                             batch=None):
    nlp2.download_file(
        'https://huggingface.co/voidful/mhubert-base/resolve/main/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', './')
    hc = HubertCode("voidful/mhubert-base", './mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', 11,
                    sampling_rate=sampling_rate,
                    chunk_sec=chunk_sec,
                    worker=worker,
                    return_diff=return_diff,
                    batch=batch)
    return hc
