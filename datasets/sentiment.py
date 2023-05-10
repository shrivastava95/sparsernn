import nltk
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
import pandas as pd

def clean_string(s):
    stop_words = set(stopwords.words('english'))
    s= re.sub(r"[^\w\s]",'',s)

    s= re.sub(r"\d",'',s)

    s = [word for word in s.split() if len(word) > 3]

    #remove stopwords
    s = [word for word in s if word not in stop_words]
    return s

class Language:
    def __init__(self,name):
        self.name = name
        self.word2index = {"<SOS>":0, "<EOS>":1, "<PAD>":2}
        self.index2word = {0:"<SOS>", 1:"<EOS>", 2:"<PAD>"}
        self.word2count = {}
        self.n_words = 3
        self.max_length = 0
    
    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)
        
        if(len(sentence) > self.max_length):
            self.max_length = len(sentence)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

def word_token_seprator(data):
    x=[]
    y=[]

    for sentence in data:
        x.append([word[0] for word in sentence])
        y.append([word[1] for word in sentence])
    
    return x, y


def word_to_token(dataset,language):
  
    dataset_token = []
   
    for sentence in dataset:
        dataset_token.append([language.word2index[word] for word in sentence])
    
    return dataset_token
    
def pad_dataset(dataset,language,left_padding=True):
    max_length = language.max_length
    if(left_padding):
        for i in range(len(dataset)):
            dataset[i] = [language.word2index["<PAD>"]]*(max_length-len(dataset[i])) + dataset[i]
    else:
        for i in range(len(dataset)):
            dataset[i] = dataset[i] + [language.word2index["<PAD>"]]*(max_length-len(dataset[i]))

    for i in range(len(dataset)):
        dataset[i] = [language.word2index["<SOS>"]] + dataset[i] + [language.word2index["<EOS>"]]

    return dataset



def create_dataset(dataset):
    x = []
    y = []
    for i in range(len(dataset)):
        x.append(clean_string(dataset.iloc[i,0]))
        y.append(dataset.iloc[i,1]=='positive')
    
    input_lang = Language('input')

    for sentence in x:
        input_lang.addSentence(sentence)

    x = word_to_token(x, input_lang)
    x = pad_dataset(x, input_lang)

    return x, y, input_lang


class SentimentDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx]).long()

def get_dataloader(x, y, batch_size=32):
    dataset = SentimentDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_train_test_dataloader(dataset, batch_size=32, test_size=0.2):
    x, y, input_lang = create_dataset(dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    train_dataloader = get_dataloader(x_train, y_train, batch_size=batch_size)
    test_dataloader = get_dataloader(x_test, y_test, batch_size=batch_size)
    return train_dataloader, test_dataloader, input_lang,None