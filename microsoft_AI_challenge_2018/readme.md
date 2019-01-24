# Microsoft AI Challenge 2018

I was able to get Rank 14 with Mean Reciprocal Rank (MRR) score of 0.7008 in evaluation 1 which used [PyTorch implementation of Bert for next sentence prediction](https://github.com/huggingface/pytorch-pretrained-BERT#4-bertfornextsentenceprediction).

[Hackathon description](https://competitions.codalab.org/competitions/20616)

 #### Problem statement:

 Given a user query and candidate passages corresponding to each, the task is to mark the most relevant passage which contains the answer to the user query.

In the data, there are 10 passages for each query out of which only one passage is correct. Therefore, only one passage is marked as label 1 and all other passages for that query are marked as label 0. Your goal is to rank the passages by scoring them such that the actual correct passage gets as high score as possible.

We provide three types of data sets to the participants -- i) the labelled train data for training your models and doing validations ii) the unlabelled eval1 data against which you submit your predictions during the contest and iii) the unlabelled eval2 data against which final predictions are submitted.

Result on eval2 dataset were used to declare winners.

## General Setup

### Windows/Linux/OSX

1) Get anaconda here and install 3.6.5 anaconda 5.2.0.

https://repo.continuum.io/archive/

2) Create an environment.
`conda create -n msai python=3.6.5 anaconda`
4) Activate environment
`conda activate msai`
3) Installation
`pip install pip install pytorch-pretrained-bert tqdm pandas`

#### Folders
- bert - Contains the solution which got the best score 0f 0.7008
- ###### notebooks
    - EDA notebooks
    - text pre-processing
    - Siamese network experiments
    - doc vector based solution

#### Usage
##### Data preparation
- Keep the data in /data
- Create training data using notebooks/etl-bert-final
##### Training
`cd bert`
`python bert-base-uncased-192-eval1/run_classifier.py`
This will save models in data/models
##### Eval1 submission
`python bert-base-uncased-192-eval1/predict.py`
##### Eval2 submission
`python bert-base-uncased-192-eval2/predict.py`
#### Ablation analysis

| model                                 | % of majority samples | length     | pre-processed | Score if non-DL | EPOCH0   | EPOCH1   | EPOCH2   | EPOCH3   |
|---------------------------------------|-----------------------|------------|---------------|-----------------|----------|----------|----------|----------|
| starting_kit\Baseline1_BM25           | 9of9                  | Full       | F             | 0.44            | NA       | NA       | NA       | NA       |
| starting_kit\Baseline2_DL             | 9of9                  | Full       | F             | 0.38            | NA       | NA       | NA       | NA       |
| notebooks/avg_word2vec_from_documents | 9of9                  | Full       | F             | 0.48            | NA       | NA       | NA       | NA       |
| notebooks/siamese2                    | 3o9                   | q:12, a:20 | T             | NA              |          |          |          | 0.48     |
| base-uncased                          | 1of9                  | 64         | T             | NA              |          | 0.675724 | 0.670795 | 0.630189 |
| base-uncased                          | 1of9                  | 64         | F             | NA              | 0.669466 | 0.676016 | 0.667268 |          |
| base-uncased                          | 9of9                  | 64         | T             | NA              | 0.663925 | 0.663081 | 0.660383 |          |
| base-uncased                          | 9of9                  | 64         | F             | NA              | 0.652431 | 0.678443 |          |          |
| base-uncased                          | 2of9                  | 64         | T             | NA              | 0.671819 | 0.658548 | 0.668686 |          |
| base-uncased                          | 3of9                  | 64         | T             | NA              | 0.673705 | 0.674967 |          |          |
| base-uncased                          | 2of9                  | 96         | T             | NA              | 0.690333 | 0.695025 | 0.690626 |          |
| base-uncased                          | 2of9                  | 110        | T             | NA              |          | 0.696411 |          |          |
| base-uncased                          | 2of9                  | 128        | T             | NA              | 0.692998 | 0.69797  | 0.689542 |          |
| base-uncased                          | 2of9                  | 150        | T             | NA              | 0.692773 | 0.699279 |          |          |
| base-uncased                          | 2of9                  | 192        | T             | NA              |          | 0.7008   |          |          |
| large-uncased                         | 1of9                  | 64         |               | 0.270695        |          | 0.26564  | 0.280375 |          |
| large-uncased                         | 2of9                  | 64         | T             |                 |          | 0.307753 |          |          |
