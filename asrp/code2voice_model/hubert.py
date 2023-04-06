import os

import nlp2

from asrp import HubertCode


def hubert_layer9_code500(chunk_sec=30, worker=20):
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin', './')
    hc = HubertCode("facebook/hubert-base-ls960", './hubert_base_ls960_L9_km500.bin', 9,
                    chunk_sec=chunk_sec,
                    worker=worker)
    return hc


def hubert_layer6_code50(chunk_sec=30, worker=20):
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin', './')
    os.rename('./km.bin', './hubert_base_ls960_L6_km50.bin')
    hc = HubertCode("facebook/hubert-base-ls960", './hubert_base_ls960_L6_km50.bin', 6,
                    chunk_sec=chunk_sec,
                    worker=worker)
    return hc


def hubert_layer6_code100(chunk_sec=30, worker=20):
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin', './')
    os.rename('./km.bin', './hubert_base_ls960_L6_km100.bin')
    hc = HubertCode("facebook/hubert-base-ls960", './hubert_base_ls960_L6_km100.bin', 6,
                    chunk_sec=chunk_sec,
                    worker=worker)
    return hc


def hubert_layer6_code200(chunk_sec=30, worker=20):
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km200/km.bin', './')
    os.rename('./km.bin', './hubert_base_ls960_L6_km200.bin')
    hc = HubertCode("facebook/hubert-base-ls960", './hubert_base_ls960_L6_km200.bin', 6,
                    chunk_sec=chunk_sec,
                    worker=worker)
    return hc
