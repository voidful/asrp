import nlp2

from asrp import Code2Speech


def hifigan_hubert_layer6_code100():
    # https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_to_speech/docs/direct_s2st_discrete_units.md
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000',
        './')
    cs = Code2Speech(tts_checkpoint='./g_00500000', vocoder='hifigan')
    return cs
