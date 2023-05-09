import torch
import torch.optim as optim


import warnings
from datasets import build_dataset

def main(args):
    #set up gpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise NotImplementedError('No GPU found')
    
    #build dataset
    train_loader, test_loader, input_lang, output_lang = build_dataset(dataset_name=args.dataset,test_size=args.test_size,batch_size=args.batch_size)

    #build model

    #build optimizer
    

    #train model

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

    #optimization
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=10, type=int)

    #misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--yaml',default=None,type=str)

    args = parser.parse_args()

    #override args with yaml
    if(args.yaml is not None):
        dict_param = load_yaml(args.yaml)
        args = override_config(args, dict_param)
    
    main(args)




    




        
