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