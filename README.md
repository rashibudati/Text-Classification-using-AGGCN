# Text-Classification-using-AGGCN

This is a relationship extraction problem which is done in the form of a classification task. I have used AGGCN which is a recent variant of Graph neural network.

## Requirements

My model was trained on GTX 1080 .  

- Python 3 (tested on 3.6.8)

- PyTorch (tested on 1.3.1)

- CUDA (tested on 8.0)

- tqdm

- unzip, wget (for downloading only)

If you wish to run the .py file follow the below steps or else run the Ipython notebook.

## Preparation
Those are the JSON files under the directory `dataset/semeval`.

First, download and unzip GloVe vectors:

```
chmod +x download.sh; ./download.sh
```
Then prepare vocabulary and initial word vectors with:

```
python3 prepare_vocab.py dataset/semeval dataset/vocab --glove_dir dataset/glove
```
This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

## Training

To train the AGGCN model, run:

```
bash train_aggcn.sh
```
Model checkpoints and logs will be saved to `./saved_models/01`.
For details on the use of other parameters, please refer to `train.py`.
## Evaluation

My pretrained model is saved under the dir saved_models/01. To run evaluation on the test set, run:

```
python3 eval.py saved_models/01 --dataset test
```
###### Note:
The dependency tags created earlier in SemEval dataset was by using Stanford Core NLP. The old version had different tags compared to new one so the earlier dataset has been converted based on the latest version of Stanford Core NLP. 
