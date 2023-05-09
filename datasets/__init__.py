from nltk.corpus import stopwords
from nltk.corpus import treebank, brown, conll2000
import pandas as pd
import nltk
import os

from .pos import get_train_test_dataloader as pos_get_train_test_dataloader
from .sentiment import get_train_test_dataloader as sentiment_get_train_test_dataloader


def build_dataset(dataset_name,test_size,batch_size):
    if(dataset_name=='pos'):
        nltk.data.path.append(os.getcwd())
        #check if nltk data is downloaded
        nltk.download('treebank', download_dir=os.getcwd())
        nltk.download('brown', download_dir=os.getcwd())
        nltk.download('conll2000', download_dir=os.getcwd())

        treebank_data = treebank.tagged_sents()
        brown_data = brown.tagged_sents()
        conll2000_data = conll2000.tagged_sents()

        dataset = treebank_data + brown_data + conll2000_data
        dataset = [sentence for sentence in dataset if len(sentence) <= 50]

        return pos_get_train_test_dataloader(dataset = dataset,batch_size=batch_size,test_size=test_size)
    elif(dataset_name=='sentiment'):
        dataset = pd.read_csv('./IMDB Dataset.csv', encoding='latin-1')
        stop_words = set(stopwords.words('english'))
        dataset = dataset[[len(sentence) <= 2500 for sentence in dataset['review']]]

        return sentiment_get_train_test_dataloader(dataset = dataset,batch_size=batch_size,test_size=test_size)



