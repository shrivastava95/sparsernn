import torch
import torch.optim as optim

from models.sparsernn import SparseRNN
import warnings
from datasets import build_dataset
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt


def main(args):
    #set up gpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise NotImplementedError('No GPU found')
    
    #build dataset
    train_loader, test_loader, input_lang, output_lang = build_dataset(dataset_name=args.dataset, test_size=args.test_size, batch_size=args.batch_size)

    #build model
    SparseRNN_kwargs = {
        'sequence_length': args.maxsize, ### edit this
        'embedding_dims': args.embedding_dims,   ### hyperparamater
        'vocab_size': input_lang.n_words,   ### edit this
        'num_classes': 1,     ### edit this (or not)
        'hidden_state_sizes': [512, 128],  ### hyperparameter
    }
    model = SparseRNN(**SparseRNN_kwargs).to(device)
    # sequences = torch.ones(batch_size, C*H*W, num_tokens).to(device)
    # model(sequences).shape

    #build optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs // args.T)



    #train model
    losses = []
    for epoch in range(args.epochs):
        for sequences, labels in tqdm(train_loader):
            labels = labels.to(device)
            scores = model(sequences)[:, -1]
            loss = criterion(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        plt.plot(losses)
        plt.title(f'train losses uptil end of epoch {epoch+1}')
        plt.show()

    



    #test model

if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser()

    def override_config(args, dict_param):
        for k, v in dict_param.items():
            if isinstance(v, dict):
                args = override_config(args, v)
            else:
                setattr(args, k, v)
        return args


    def load_yaml(path):
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return d

    #dataset
    parser.add_argument('--dataset', default='sentiment', choices=['pos', 'sentiment'])

    #models
    parser.add_argument('--model', default='sparsernn', choices=['sparsernn','rnn','sparsernn_seqtoseq','rnn_seqtoseq'])
    parser.add_argument('--maxsize', default=512, type=int)           # maximum sequence length to which sequences are padded.
    parser.add_argument('--embedding_dims', default=64, type=int)     # embedding size for the model internals

    #optimization
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--T', default=4, type=int)

    #misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cycles',default=None,type=str)

    args = parser.parse_args()

    #override args with yaml
    if(args.yaml is not None):
        dict_param = load_yaml(args.yaml)
        args = override_config(args, dict_param)
    
    main(args)




    




        
