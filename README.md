# asrp

ASR text preprocessing utility

## install

`pip install asrp`

## usage - preprocess

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

## usage - evaluation

```python
import asrp

targets = ['HuggingFace is great!', 'Love Transformers!', 'Let\'s wav2vec!']
preds = ['HuggingFace is awesome!', 'Transformers is powerful.', 'Let\'s finetune wav2vec!']
print("chunk size WER: {:2f}".format(100 * asrp.chunked_wer(targets, preds, chunk_size=None)))
print("chunk size CER: {:2f}".format(100 * asrp.chunked_cer(targets, preds, chunk_size=None)))
```

## usage - hubertcode

```python
import asrp

hc = asrp.HubertCode("facebook/hubert-large-ll60k", './km_feat_100_layer_20', 20)
hc('voice file path')
```

## usage - code2speech

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

### usage - liveASR

* modify from https://github.com/oliverguhr/wav2vec2-live

```python
from asrp.live import LiveHFSpeech

english_model = "voidful/wav2vec2-xlsr-multilingual-56"
asr = LiveHFSpeech(english_model, device_name="default")
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