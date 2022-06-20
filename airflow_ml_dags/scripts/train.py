import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-dd", "--data_dir", 
                        dest="data_dir", 
                        type=str, 
                        default=None, 
                        help="path to data directory",
                        )
    parser.add_argument("-od", "--output_dir", 
                        dest="output_dir", 
                        type=str, 
                        default=None, 
                        help="path to out directory where to put model",
                        )
    return parser.parse_args()

def train(data_dir: str, output_dir: str):
    X_train = pd.read_csv(data_dir + 'data_train.csv')
    y_train = pd.read_csv(data_dir + 'target_train.csv')
    model = LogisticRegression()
    model.fit(X_train, y_train['target'])
    with open(output_dir + 'model.pkl', 'wb') as file:
        pickle.dump(model, file)
    

def main(args):
    train(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
