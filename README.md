# NMT_MASS

Course project of Northwestern University COMP_SCI 496: Statistical Machine Learning.

## Build Setup

```py
# download pre-trained model
wget https://modelrelease.blob.core.windows.net/mass/zhen_mass_pre-training.pt -P ./MASS/model

# install python libraries
pip install -r requirements.txt

# run the server
python app.py
```

## Project Structure

```
- NMT-MASS/
  ├─ MASS/
  |  ├─ __init__.py
  |  ├─ mass
  |  |  ├─ __init__.py
  |  |  ├─ masked_language_pair_dataset.py
  |  |  ├─ noisy_language_pair_dataset.py
  |  |  ├─ xmasked_seq2seq.py
  |  |  └─ xtransformer.py
  |  ├─ model
  |  |  ├─ bpe
  |  |  |  ├─ all.en.bpe.codes
  |  |  |  └─ all.zh.bpe.codes
  |  |  ├─ data
  |  |  |  ├─ test
  |  |  |  |  ├─ valid.en
  |  |  |  |  └─ valid.zh
  |  |  |  ├─ dict.en.txt
  |  |  |  └─ dict.zh.txt
  |  |  └─ zhen_mass_pre-training.ot
  |  └─ nmt.py
  ├─ requirements.txt
  └─ app.py
```

## Reference
[MAsked Sequence to Sequence](https://github.com/microsoft/MASS)
