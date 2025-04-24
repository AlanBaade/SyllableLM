# SyllableLM
Official Public Code for "SyllableLM: Learning Coarse Semantic Units for Speech Language Models"

Paper: [https://arxiv.org/abs/2410.04029](https://arxiv.org/abs/2410.04029)

ICLR 2025

[Demo Audios](https://syllablelmanonymous.github.io/SyllableLMAnonymous/)


## Setup:

```bash
conda create -n syllablelm python=3.9
conda activate syllablelm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu115

pip install omegaconf
pip install timm
```



## SylBoost:

### Checkpoints

| SylBoost | Model                                                                                        | KMeans                                                                                              | Agglomerative Clustering                                                                             |
|----------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 8.33Hz   | [Model](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_833Hz.pth) | [KMeans](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_833Hz_kmeans.npy) | [Agglom](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_833Hz_agglom.npy) |
| 6.25Hz   | [Model](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_625Hz.pth) | [KMeans](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_625Hz_kmeans.npy) | [Agglom](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_625Hz_agglom.npy) |
| 5.0Hz    | [Model](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_500Hz.pth) | [KMeans](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_500Hz_kmeans.npy) | [Agglom](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SylBoost_500Hz_agglom.npy) |


### Usage

SylBoost inference and efficient extraction code in ``extract_units.py``

People have had trouble setting up Data2Vec2 so I copied it and stripped it. No Fairseq reqired! 

```python
sylboost_reader = SylBoostFeatureReader(
        '/path/to/model.pt'
        '/path/to/kmeans.npy',
        '/path/to/agglom.npy',
        '8.33Hz',  # '6.25Hz', '5.0Hz'
    )
```

## SyllableLM:

### Checkpoints

| SyllableLM                    | Model                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------|
| 6.25Hz Base                   | [Model](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SyllableLM_base_625Hz.pt)       |
| 6.25Hz Large                  | [Model](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SyllableLM_large_625Hz.pt)      |
| 6.25Hz Interleaved Vocoder LM | [Model](https://www.cs.utexas.edu/~harwath/model_checkpoints/syllable_lm/SyllableLM_vocoder_lm_625Hz.pt) |

### Usage

Todo: migrate code over and facilitate twist dependency.

## Resynthesis:

Todo

## Continuation Pipeline:

Todo

## LossPred:

This will be provided as-is

## SylBoost training:

This will be provided as-is

## SyllableLM training:

This is standard language model training and will be provided as is.
