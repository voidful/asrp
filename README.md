# asrp

ASR text preprocessing utility

## install

`pip install asrp`

## Preprocess

input: dictionary, with key `sentence`    
output: preprocessed result, inplace handling.

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

```python
import asrp

targets = ['HuggingFace is great!', 'Love Transformers!', 'Let\'s wav2vec!']
preds = ['HuggingFace is awesome!', 'Transformers is powerful.', 'Let\'s finetune wav2vec!']
print("chunk size WER: {:2f}".format(100 * asrp.chunked_wer(targets, preds, chunk_size=None)))
print("chunk size CER: {:2f}".format(100 * asrp.chunked_cer(targets, preds, chunk_size=None)))
```

## Speech to Hubert code

```python
import asrp
import nlp2

nlp2.download_file(
    'https://huggingface.co/voidful/mhubert-base/resolve/main/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', './')
hc = asrp.HubertCode("voidful/mhubert-base", './mhubert_base_vp_en_es_fr_it3_L11_km1000.bin', 11,
                     chunk_sec=30,
                     worker=20)
hc('voice file path')
```

## Hubert code to speech

```python
import asrp

code = []  # discrete unit
# download tts checkpoint and waveglow_checkpint from https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/unit2speech
cs = asrp.Code2Speech(tts_checkpoint='./tts_checkpoint_best.pt', waveglow_checkpint='waveglow_256channels_new.pt')
cs(code)

# play on notebook
import IPython.display as ipd

ipd.Audio(data=cs(code), autoplay=False, rate=cs.sample_rate)
```

### Speech Enhancement

Denoiser copied
from [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_synthesis/preprocessing/denoiser)

```python
from asrp import SpeechEnhancer

ase = SpeechEnhancer()
print(ase('./test/xxx.wav'))
```

### usage - liveASR

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

### usage - liveASR - whisper

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