# HMEs-Recognition
This is the simple implementation for the paper "[Hierarchical Multi-Label Ensemble for Image Classification](https://arxiv.org/abs/1901.06763)".
The code is using lightning framework to simplify the training process. This code does not completely reproduce the data processing part in the paper, but it can be used as a baseline for further research.

## Requirements
Usually, you need to make sure your version greater or equal to the following versions:
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
python main.py --test --checkpoint_path checkpoints/best-checkpoint-exp-rate.ckpt
```
The model will be loaded from the folder `checkpoints/`. You can also change the hyperparameters in `main.py`.

You may also use tensorboard to visualize the training process. The `lightning_logs` folder is logdir


## Results
The results are shown below. The results are not exactly the same as the results in the paper.

For 2013 dataset:
```
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        accuracy            0.6508222818374634
        f1_score            0.7515795230865479
        precision           0.9520794749259949
         recall             0.6508222818374634
      test_exp_rate         0.4946236312389374
  test_exp_rate_less_1      0.6094470024108887
  test_exp_rate_less_2      0.6571140885353088
  test_exp_rate_less_3      0.6824597120285034
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
For 2014 dataset:
```
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        accuracy            0.6697215437889099
        f1_score            0.7698380947113037
        precision           0.9536325931549072
         recall             0.6697215437889099
      test_exp_rate         0.5645161271095276
  test_exp_rate_less_1      0.6030551791191101
  test_exp_rate_less_2      0.6297301650047302
  test_exp_rate_less_3       0.659972071647644
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
For 2016 dataset:
```
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        accuracy            0.6802293062210083
        f1_score            0.7791554927825928
        precision           0.9568066596984863
         recall             0.6802293062210083
      test_exp_rate         0.5624678730964661
  test_exp_rate_less_1      0.6102108955383301
  test_exp_rate_less_2      0.6357060074806213
  test_exp_rate_less_3      0.6556712985038757
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```