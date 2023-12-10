# HMEs-Recognition
This is the simple implementation for the paper "[Hierarchical Multi-Label Ensemble for Image Classification](https://arxiv.org/abs/1901.06763)".
The code is using lightning framework to simplify the training process. This code does not completely reproduce the data processing part in the paper, but it can be used as a baseline for further research.

## Requirements
- Python 3.8
- pytroch 1.12.1
- lightning 2.1.2
- torchvision 0.13.1

## Dataset
The dataset is CROHME dataset, which is a handwritten math expression dataset. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/jungomi/chrome-png). The dataset is preprocessed by the code in [here](https://github.com/jungomi/math-formula-recognition).

Please put the dataset in the folder `data/` with the following structure:
```
data/
    gt_split/
        train.tsv
        validation.tsv
    train/
        ...
    test/
        2013/
            *.png
        2014/
            *.png
        2016/
            *.png
    tokens.tsv
    ...
```

## Model
The model is following the structure in the paper. The models are defined in `model`. The model is trained with the cross entropy loss.

You may download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1g6LnaHuJkPI2z5X7Qh4Ms2HdZpSvlSxG?usp=drive_link
)
## Training
You can train the model with the following command:
```bash
python main.py --train --early_stop --checkpoint
```
The model will be saved in the folder `checkpoints/`. You can also change the hyperparameters in `main.py`.

## Evaluation
You can evaluate the model with the following command:
```bash
python main.py
```
The model will be loaded from the folder `checkpoints/`. You can also change the hyperparameters in `main.py`.

## Results
TODO