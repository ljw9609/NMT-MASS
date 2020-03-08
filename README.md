# NMT_MASS

Course project of Northwestern University COMP_SCI 496: Statistical Machine Learning.

## Problem Statement

In this project, we want to develop a English to Chinese bidirectional machine translator. The project includes 3 parts:

  + [NMT-MASS](https://github.com/ljw9609/NMT-MASS): MAsked Seq-to-Seq Model

  + [NMT-FCONV](https://github.com/ljw9609/NMT-FCONV): Fully CONVolutional seq-to-seq Model

  + [NMT-UI](https://github.com/ljw9609/NMT-UI): User Interface

### Input & Output

+ Input: Sentences or paragraphs in source language. (eg: Slowly and not without struggle, America began to listen.)

+ Output: Sentences or paragraphs in target language. (eg: 美国缓慢地开始倾听，但并非没有艰难曲折。)

## Core Model

The project includes 2 core models, [MASS](https://github.com/microsoft/MASS) and [FCONV](https://github.com/pytorch/fairseq/tree/master/examples/conv_seq2seq). MASS is a pre-trained model provided by Microsoft. FCONV has no pre-trained model provided and need to be trained from scratch.

## Deliverables

+ Pre-trained model [Download](https://modelrelease.blob.core.windows.net/mass/zhen_mass_pre-training.pt)
+ Model Inference [View](https://github.com/ljw9609/NMT-MASS/blob/master/MASS/nmt.py)
+ Web API [View](https://github.com/ljw9609/NMT-MASS/blob/master/src/app.py)
+ Dockerfile [View](https://github.com/ljw9609/NMT-MASS/blob/master/Dockerfile)
+ Docker Image [View](https://hub.docker.com/repository/docker/ljw96/nmt-mass)

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
  ├─ src/
  |  └─app.py
  └─ requirements.txt
```

## Model Training

Microsoft has already provided the pre-trained model, so we didn't retrain it. 

## Usage

### Model Inference

```py
en2zh = Translator()
en2zh.initialize(
  data_dir,
  model, 
  user_dir, 
  task='xmasked_seq2seq',
  s_lang='en', t_lang='zh',
  bpe_codes=bpe_codes, beam=5
)
en2zh.translate('Slowly and not without struggle, America began to listen.')
```

### Start server

#### 1. Run on local

```py
# download pre-trained model
wget https://modelrelease.blob.core.windows.net/mass/zhen_mass_pre-training.pt -P ./MASS/model

# install python libraries
pip install -r requirements.txt

# run the server
gunicorn --config ./conf/gunicorn_config.py src:app
```

#### 2. Run in Docker

```sh
# build image
docker build -t nmt-mass .

# start a container
docker run -p 4869:8000 --name mass nmt-mass
```

### Web API

#### HTTP Request

```json
POST /translate
Host: YOUR_SERVER_ADDRESS
Body: {
  's_lang': 'en',
  't_lang': 'zh',
  's_text': 'Slowly and not without struggle, America began to listen.'
}
```

#### HTTP Response

```json
{
  's_text': 'Slowly and not without struggle, America began to listen.',
  't_text': '美国缓慢地开始倾听，但并非没有艰难曲折。'
}
```

## Reference
[MAsked Sequence to Sequence](https://github.com/microsoft/MASS)
