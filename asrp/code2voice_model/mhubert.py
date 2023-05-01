import asrp
import nlp2


# https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_to_speech/docs/textless_s2st_real_data.md
def hifigan_mhubert_en_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000',
        './', 'hifigan_mhubert_en_layer11_code1000_g_00500000')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json',
        './', 'hifigan_mhubert_en_layer11_code1000_config.json')
    cs = asrp.Code2Speech(tts_checkpoint='./hifigan_mhubert_en_layer11_code1000_g_00500000',
                          model_cfg='hifigan_mhubert_en_layer11_code1000_config.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_es_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000',
        './', 'hifigan_mhubert_es_layer11_code1000_g_00500000')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/config.json',
        './', 'hifigan_mhubert_es_layer11_code1000_config.json')
    cs = asrp.Code2Speech(tts_checkpoint='./hifigan_mhubert_es_layer11_code1000_g_00500000',
                          model_cfg='hifigan_mhubert_es_layer11_code1000_config.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_fr_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000',
        './', 'hifigan_mhubert_fr_layer11_code1000_g_00500000')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/config.json',
        './', 'hifigan_mhubert_fr_layer11_code1000_config.json')
    cs = asrp.Code2Speech(tts_checkpoint='./hifigan_mhubert_fr_layer11_code1000_g_00500000',
                          model_cfg='hifigan_mhubert_fr_layer11_code1000_config', vocoder='hifigan')
    return cs


# https://github.com/facebookresearch/fairseq/tree/ust/examples/speech_matrix
def hifigan_mhubert_de_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_de.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_de.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_de.pt', model_cfg='config_de.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_nl_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_nl.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_nl.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_nl.pt', model_cfg='config_nl.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_fi_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_fi.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_fi.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_fi.pt', model_cfg='config_fi.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_hu_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_hu.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_hu.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_hu.pt', model_cfg='config_hu.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_et_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_et.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_et.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_et.pt', model_cfg='config_et.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_it_layer11_code800():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_it.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_it.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_it.pt', model_cfg='config_it.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_pt_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_pt.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_pt.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_pt.pt', model_cfg='config_pt.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_ro_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_ro.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_ro.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_ro.pt', model_cfg='config_ro.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_cs_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_cs.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_cs.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_cs.pt', model_cfg='config_cs.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_pl_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_pl.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_pl.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_pl.pt', model_cfg='config_pl.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_hr_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_hr.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_hr.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_hr.pt', model_cfg='config_hr.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_lt_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_lt.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_lt.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_lt.pt', model_cfg='config_lt.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_sk_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_sk.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_sk.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_sk.pt', model_cfg='config_sk.json', vocoder='hifigan')
    return cs


def hifigan_mhubert_sl_layer11_code1000():
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_sl.pt',
        './')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_sl.json',
        './')
    cs = asrp.Code2Speech(tts_checkpoint='./vocoder_sl.pt', model_cfg='config_sl.json', vocoder='hifigan')
    return cs

# todo: hokkien https://github.com/facebookresearch/fairseq/tree/ust/examples/hokkien
