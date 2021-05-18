# asrp

ASR text preprocessing utility

## install

`pip install asrp`

## usage

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