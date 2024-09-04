STBA: Span-Based Tagging Scheme with Biaffine Attention for Enhanced Aspect Sentiment Triplet Extraction is based on 


# STAGE: Span Tagging and Greedy Inference scheme for Aspect Sentiment Triplet Extraction

This repository contains Pytorch implementation for "STAGE: Span Tagging and Greedy Inference Scheme for Aspect Sentiment Triplet Extraction" (AAAI 2023) ([AAAI Version](https://ojs.aaai.org/index.php/AAAI/article/view/26547) and [Arxiv Version](https://arxiv.org/abs/2211.15003))

## 1. Requirements

We conduct our experiments on Nvidia GeForce 3090 GPU, with CUDA version 11.6 and PyTorch v1.10.1. 

To reproduce experimental environment.
```
conda create -n STAGE python=3.9
conda activate STAGE
python -m pip install -r requirements.txt
```

## 2. Data

We use `ASTE-Data-V2-EMNLP2020` from https://github.com/xuuuluuu/SemEval-Triplet-data.git
(widely-used datasets in ASTE task)

The data dir should be  ``data/ASTE-Data-V2-EMNLP2020`` (*or* , set the correct ***dataset_dir*** parameter during training or predicting)


## 3. Train

To reproduce our *best* test $F_1$ performance on four datasets:
```
python run.py
```
Best $F_1$ scores are shown in `logs/best_score.txt` when running on our environment. 

We also provide our training log in `logs/best_training.log`. Please ignore the time information as another tasks were also running at the same time.

## 4. Evaluate

Change model_path, dataset, version variants in ``predict.py'' and run:
```
python predict.py
```
We provide the output file `logs/best_16res_3D_predict.log`


   PAGES      = {13174-13182} 
}
```
