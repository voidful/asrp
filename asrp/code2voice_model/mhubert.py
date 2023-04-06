import nlp2

from asrp import HubertCode


def mhubert_layer11_code1000(chunk_sec=30, worker=20):
    nlp2.download_file(
        'https://huggingface.co/voidful/mhubert-base/resolve/main/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', './')
    hc = HubertCode("voidful/mhubert-base", './mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', 11,
                    chunk_sec=chunk_sec,
                    worker=worker)
    return hc
