
""" Train or Eval the model for handwriting recognintion
Usage:
    run.py [-h]
            [--train-data=<train>]
            [--dev-data=<train>]
            [--test-data=<test>]
            [--initial-model=<model>]
            [--train-batch-size=<N>]
            [--test-batch-size=<M>]
            [--epochs=<K>]
            [--learning-rate=<R>]
            [--learning-rate-factor=<F>]
            [--gpu=<G>]
            [--model-dir=<dir>]
            --target=<target>

Options:
    -h, --help                  show this help message and exit
    --train-data=<train>        training data in json format
    --dev-data=<dev>            validation data in json format
    --test-data=<test>          test data in json format
    --initial-model=<model>     model path for initializing
    --train-batch-size=<N>      batch size for training [default: 16]
    --test-batch-size=<M>       batch size for test [default: 1]
    --epochs=<K>                number of training epoch [default: 30]
    --learning-rate=<R>         learning rate [default: 0.0001]
    --learning-rate-factor=<F>  attenuation rate of learning rate [default: 0.5]
    --gpu=<G>                   whether to use GPU or not [default: False]
    --model-dir=<dir>           output direcrory [default: output]
    --traget=<target>           target labels in joson format
"""
from docopt import docopt
import json
import torch
import os

from model.cnn_lstm_model import CnnLstmModel
from model.cnn_lstm_e2e_model import CnnLstmE2EModel
from train import train
from test import evaluate


def main(args):
    print(args)
    targets = '_'
    with open(args['--target']) as fi:
        targets = json.load(fi)
        targets = ''.join(targets)
    model = CnnLstmE2EModel(targets=targets)
    device = torch.device('cuda' if args['--gpu']=='True' else 'cpu')
    model = model.to(device)
    if not os.path.exists(args['--model-dir']):
        os.makedirs(args['--model-dir'])
    if args['--initial-model']:
        print(f'Initialize the model: {args["--initial-model"]}')
        model.load_state_dict(torch.load(args['--initial-model']))
    if args['--train-data']:
        print('-- Start Training')
        train(model=model,
                train_data=args['--train-data'],
                dev_data=args['--dev-data'],
                targets=targets,
                output_dir=args['--model-dir'],
                batch_size=int(args['--train-batch-size']),
                n_epochs=int(args['--epochs']),
                learning_rate=float(args['--learning-rate']),
                learning_rate_factor=float(args['--learning-rate-factor']),
                device=device)
        print('-- Finish Training')
    if args['--test-data']:
        best_model = os.path.join(args['--model-dir'], f'{model.__class__.__name__}_checkpoint_best.pth')
        if os.path.exists(best_model):
            print(f'Load the model: {best_model}')
            model.load_state_dict(torch.load(best_model))
        model = model.to('cpu')
        print('-- Start Test')
        evaluate(model=model,\
                    test_data=args['--test-data'],
                    targets=targets,
                    output_dir=args['--model-dir'],
                    batch_size=int(args['--test-batch-size']))
        print('-- Finish Test')


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
        
        
