# sparsernn

An efficient RNN-inspired sequence modelling architecture called SparseRNN, which employs dilation to reduce the cost of dense-recurrence without compromising on performance.

This was made as the final course project for the Deep Learning\[DL2023\] course at Indian Institute of Technology, Jodhpur. 

## Research Report & Demo Video

The [research report(link)](sparsernn-report.pdf) contains in-depth explanations of our research and the results obtained on three benchmarks.

Refer to the [demo video(link)](sparseRNN-demo-video.mp4) for a barebones presentation and commentary on our work.


## Requirements
- Python >= 3.6
- PyTorch >= 1.9
- Torchvision >= 0.10
- wandb >= 0.12.1
- cuda >= 10.2

## Datasets
- [Penn Treebank](https://www.kaggle.com/datasets/nltkdata/penn-tree-bank)
- [conll2000](https://www.kaggle.com/datasets/nltkdata/conll-corpora)
- [brown](https://www.kaggle.com/nltkdata/brown-corpus)
- [MDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Tasks
- part-of-speech tagging. Datasets used-
    - Penn Treebank
    - conll2000
    - brown
- Sentiment Analysis. Dataset used-
    - MDB Dataset of 50K Movie Reviews

## models trained
- SparseRNN
- LSTM
- RNN

## Usage
To run a model on for a task, run the following command:
```
python main.py --yaml <path to yaml file in config folder>
```


For example, to run sparsernn for sentiment analysis, run the command:
``` 
python main.py --yaml ./config/sentiment/sparsernn.yaml
```

## Contributors
- [Jahnab Dutta](mailto:dutta.4@iitj.ac.in)
- [Ishaan Shrivastava](mailto:shrivastava.9@iitj.ac.in)
- [Vikash Yadav](mailto:yadav.41@iitj.ac.in)