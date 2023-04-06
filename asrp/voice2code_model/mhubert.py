import nlp2

import asrp


# https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_to_speech/docs/textless_s2st_real_data.md
def hifigan_mhubert_en_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./g_00500000', vocoder='hifigan')
    return cs


def hifigan_mhubert_es_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./g_00500000', vocoder='hifigan')
    return cs


def hifigan_mhubert_fr_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./g_00500000', vocoder='hifigan')
    return cs


# https://github.com/facebookresearch/fairseq/tree/ust/examples/speech_matrix
def hifigan_mhubert_de_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_de.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_de.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_nl_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_nl.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_nl.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_fi_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_fi.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_fi.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_hu_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_hu.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_hu.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_et_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_et.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_et.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_it_layer11_code800():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_it.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_it.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_pt_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_pt.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_pt.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_ro_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_ro.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_ro.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_cs_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_cs.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_cs.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_pl_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_pl.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_pl.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_hr_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_hr.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_hr.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_lt_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_lt.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_lt.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_sk_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_sk.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_sk.pt', vocoder='hifigan')
    return cs


def hifigan_mhubert_sl_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_sl.pt',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_sl.pt', vocoder='hifigan')
    return cs

# todo: hokkien https://github.com/facebookresearch/fairseq/tree/ust/examples/hokkien