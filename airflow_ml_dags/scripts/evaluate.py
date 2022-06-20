import os
import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pickle

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-dd", "--data_dir", 
                        dest="data_dir", 
                        type=str, 
                        default=None, 
                        help="path to data directory",
                        )
    parser.add_argument("-md", "--model_dir", 
                        dest="model_dir", 
                        type=str, 
                        default=None, 
                        help="path to directory where model is located",
                        )
    parser.add_argument("-od", "--output_dir", 
                        dest="output_dir", 
                        type=str, 
                        default=None, 
                        help="path to directory where to put results",
                        )
    return parser.parse_args()

def accuracy(y_true, pred):
    return np.sum(y_true == pred) / y_true.shape[0]

def evaluate(data_dir: str, model_dir: str, output_dir: str):
    X = pd.read_csv(data_dir + 'data_val.csv')
    y = pd.read_csv(data_dir + 'target_val.csv')
    y = y['target']
    with open(model_dir + 'model.pkl', 'rb') as file:
        model = pickle.load(file)
    pred = model.predict(X)
    acc = accuracy(y, pred)
    res = {
        'model': model_dir + 'model.pkl',
        'accuracy': acc,
    }
    with open(output_dir + 'metrics.json', 'w') as file:
        json.dump(res, file)
    

def main(args):
    evaluate(data_dir=args.data_dir, model_dir=args.model_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
