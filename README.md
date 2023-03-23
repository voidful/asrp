# ASRP: Automatic Speech Recognition Preprocessing Utility

ASRP is a python package that offers a set of tools to preprocess and evaluate ASR (Automatic Speech Recognition) text.
The package also provides a speech-to-text transcription tool and a text-to-speech conversion tool. The code is
open-source and can be installed using pip.

Key Features

- [Preprocess ASR text with ease](#preprocess)
- [Evaluate ASR output quality](#Evaluation)
- [Transcribe speech to Hubert code](#speech-to-discrete-unit)
- [Convert unit code to speech](#discrete-unit-to-speech)
- [Enhance speech quality with a noise reduction tool](#speech-enhancement)
- [LiveASR tool for real-time speech recognition](#liveasr---huggingfaces-model)
- [Speaker Embedding Extraction (x-vector/d-vector)](#speaker-embedding-extraction---x-vector)

## install

`pip install asrp`

## Preprocess

ASRP offers an easy-to-use set of functions to preprocess ASR text data.   
The input data is a dictionary with the key 'sentence', and the output is the preprocessed text.     
You can either use the fun_en function or use dynamic loading. Here's how to use it:

```python
import asrp

batch_data = {
    'sentence': "I'm fine, thanks."
}
asrp.fun_en(batch_data)
```

dynamic loading

```python
import asrp

batch_data = {
    'sentence': "I'm fine, thanks."
}
preprocessor = getattr(asrp, 'fun_en')
preprocessor(batch_data)
```

## Evaluation

ASRP provides functions to evaluate the output quality of ASR systems using     
the Word Error Rate (WER) and Character Error Rate (CER) metrics.   
Here's how to use it:

```python
import asrp

targets = ['HuggingFace is great!', 'Love Transformers!', 'Let\'s wav2vec!']
preds = ['HuggingFace is awesome!', 'Transformers is powerful.', 'Let\'s finetune wav2vec!']
print("chunk size WER: {:2f}".format(100 * asrp.chunked_wer(targets, preds, chunk_size=None)))
print("chunk size CER: {:2f}".format(100 * asrp.chunked_cer(targets, preds, chunk_size=None)))
```

## Speech to Discrete Unit

```python
import asrp
import nlp2

# https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_to_speech/docs/textless_s2st_real_data.md
# https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/ulm
nlp2.download_file(
    'https://huggingface.co/voidful/mhubert-base/resolve/main/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', './')
hc = asrp.HubertCode("voidful/mhubert-base", './mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', 11,
                     chunk_sec=30,
                     worker=20)
hc('voice file path')
```

## Discrete Unit to speech

```python
import asrp

code = []  # discrete unit
# https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/unit2speech
# https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_to_speech/docs/textless_s2st_real_data.md
cs = asrp.Code2Speech(tts_checkpoint='./tts_checkpoint_best.pt', waveglow_checkpint='waveglow_256channels_new.pt')
cs(code)

# play on notebook
import IPython.display as ipd

ipd.Audio(data=cs(code), autoplay=False, rate=cs.sample_rate)
```

mhubert English hifigan vocoder example

```python
import asrp
import nlp2
import IPython.display as ipd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
nlp2.download_file(
    'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000',
    './')


tokenizer = AutoTokenizer.from_pretrained("voidful/mhubert-unit-tts")
model = AutoModelForSeq2SeqLM.from_pretrained("voidful/mhubert-unit-tts")
model.eval()
cs = asrp.Code2Speech(tts_checkpoint='./g_00500000', vocoder='hifigan')

inputs = tokenizer(["The quick brown fox jumps over the lazy dog."], return_tensors="pt")
code = tokenizer.batch_decode(model.generate(**inputs,max_length=1024))[0]
code = [int(i) for i in code.replace("</s>","").replace("<s>","").split("v_tok_")[1:]]
print(code)
ipd.Audio(data=cs(code), autoplay=False, rate=cs.sample_rate)

```

## Speech Enhancement

ASRP also provides a tool to enhance speech quality with a noise reduction tool.  
from https://github.com/facebookresearch/fairseq/tree/main/examples/speech_synthesis/preprocessing/denoiser

```python
from asrp import SpeechEnhancer

ase = SpeechEnhancer()
print(ase('./test/xxx.wav'))
```

## LiveASR - huggingface's model

* modify from https://github.com/oliverguhr/wav2vec2-live

```python
from asrp.live import LiveSpeech

english_model = "voidful/wav2vec2-xlsr-multilingual-56"
asr = LiveSpeech(english_model, device_name="default")
asr.start()

try:
    while True:
        text, sample_length, inference_time = asr.get_last_text()
        print(f"{sample_length:.3f}s"
              + f"\t{inference_time:.3f}s"
              + f"\t{text}")

except KeyboardInterrupt:
    asr.stop()
```

## LiveASR - whisper's model

```python
from asrp.live import LiveSpeech

whisper_model = "tiny"
asr = LiveSpeech(whisper_model, vad_mode=2, language='zh')
asr.start()
last_text = ""
while True:
    asr_text = ""
    try:
        asr_text, sample_length, inference_time = asr.get_last_text()
        if len(asr_text) > 0:
            print(asr_text, sample_length, inference_time)
    except KeyboardInterrupt:
        asr.stop()
        break

```

## Speaker Embedding Extraction - x vector

from https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.Xvector.html

```python
from asrp.speaker_embedding import extract_x_vector

extract_x_vector('./test/xxx.wav')
```

## Speaker Embedding Extraction - d vector

from https://github.com/yistLin/dvector

```python
from asrp.speaker_embedding import extract_d_vector

extract_d_vector('./test/xxx.wav')
```
