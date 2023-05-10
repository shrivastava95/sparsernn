import torch
import torch.optim as optim

from models.sparsernn import SparseRNN
import warnings
from datasets import build_dataset
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt

import wandb
import numpy as np

type_scheduler = 'warmup'

def main(args):
    #set up gpu
    hidden_state_sizes = [512, 128] if args.dataset == 'sentiment' else [512, 128]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise NotImplementedError('No GPU found')
    
    #build dataset
    train_loader, test_loader, input_lang, output_lang = build_dataset(dataset_name=args.dataset, test_size=args.test_size, batch_size=args.batch_size)

    #build model
    SparseRNN_kwargs = {
        'sequence_length': input_lang.max_length + 2, ### edit this
        'embedding_dims': args.embedding_dims,   ### hyperparamater
        'vocab_size': input_lang.n_words,   ### edit this
        'num_classes': 2,     ### edit this (or not)
        'hidden_state_sizes': hidden_state_sizes,  ### hyperparameter
    }
    model = SparseRNN(**SparseRNN_kwargs).to(device)
    # sequences = torch.ones(batch_size, C*H*W, num_tokens).to(device)
    # model(sequences).shape

    #build optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if type_scheduler == 'warmup':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1., (epoch+1) / args.warmup_steps))
    elif type_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs // args.T)

    wandb.init(
        project="DL PROJECT",
        name = f"SparseRNN {args.dataset.capitalize()}",
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "embedding_dims": args.embedding_dims,
            "hidden_state_sizes": hidden_state_sizes,
            "depth": len(hidden_state_sizes),
        },
        settings=wandb.Settings(start_method="thread")
    )
    config = wandb.config

    #train model
    losses = []
    for epoch in range(args.epochs):
        model.train()
        for i, (sequences, labels) in enumerate(tqdm(train_loader)):
            labels = labels.to(device)
            sequences = sequences.to(device)
            if args.dataset == 'sentiment':
                scores = model(sequences)[:, -1]
                preds = scores.argmax(dim=1)
                correct = int(sum(preds == labels))
                total = int(labels.shape[0])
                accuracy = float(correct / total) * 100
                loss = criterion(scores, labels)
                if loss.item() > 1:
                    continue
            elif args.dataset == 'pos':
                scores = model(sequences)
                labels = labels.reshape9([-1])
                scores = model(sequences).reshape([-1, 2])
                preds = scores.argmax(dim=1)
                correct = int(sum(preds == labels))
                total = int(labels.shape[0])
                accuracy = float(correct / total) * 100
                loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            wandb.log({"Train Loss":loss.item(),"Train Accuracy":accuracy}, step=epoch*len(train_loader) + i)
            
            if type_scheduler == 'warmup':
                scheduler.step()
            elif type_scheduler == 'cosine':
                scheduler.step(epoch + i / len(train_loader))

        model.eval()
        test_accuracies = []
        test_losses = []
        test_sizes = []
        for i, (sequences, labels) in enumerate(tqdm(test_loader)):
            labels = labels.to(device)
            sequences = sequences.to(device)
            if args.dataset == 'sentiment':
                scores = model(sequences)[:, -1]
                preds = scores.argmax(dim=1)
                correct = int(sum(preds == labels))
                total = int(labels.shape[0])
                accuracy = float(correct / total) * 100
                loss = criterion(scores, labels).detach().item()
                size = labels.shape[0]
                test_accuracies.append(accuracy)
                test_losses.append(loss)
                test_sizes.append(size)
            elif args.dataset == 'pos':
                scores = model(sequences)
                labels = labels.reshape9([-1])
                scores = model(sequences).reshape([-1, 2])
                preds = scores.argmax(dim=1)
                correct = int(sum(preds == labels))
                total = int(labels.shape[0])
                accuracy = float(correct / total) * 100
                loss = criterion(scores, labels).item()
                size = labels.shape[0]
                test_accuracies.append(accuracy)
                test_losses.append(loss)
                test_sizes.append(size)
    
        test_accuracies = np.array(test_accuracies)
        test_losses = np.array(test_losses)
        test_sizes = np.array(test_sizes)
        test_accuracies = sum( test_accuracies * test_sizes ) / sum(test_sizes)
        test_losses     = sum( test_losses     * test_sizes ) / sum(test_sizes)
        wandb.log((test_losses, test_accuracies))
        torch.save(
            {
                'state_dict': model.state_dict(),
                'model_kwargs': SparseRNN_kwargs
            },
            f'results_{args.dataset}.pt'
        )
    wandb.finish()



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
    # parser.add_argument('--maxsize', type=int)           # maximum sequence length to which sequences are padded.
    parser.add_argument('--embedding_dims', default=64, type=int)     # embedding size for the model internals

    #optimization
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--T', default=3, type=int)

    #misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cycles',default=None,type=str)
    parser.add_argument('--yaml', default=None, type=str)

    args = parser.parse_args()

    #override args with yaml
    if(args.yaml is not None):
        dict_param = load_yaml(args.yaml)
        args = override_config(args, dict_param)
    
    main(args)




    




        
