import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-dd", "--data_dir", 
                        dest="data_dir", 
                        type=str, 
                        default=None, 
                        help="path to data directory",
                        )
    parser.add_argument("-vs", "--validation_size", 
                        dest="validation_size", 
                        type=float, 
                        default=None, 
                        help="validation size, must be float between 0 and 1",
                        )
    return parser.parse_args()

def split(data_dir: str, val_size: float):
    df = pd.read_csv(data_dir + 'data.csv')
    target = pd.read_csv(data_dir + 'target.csv')
    X_train, X_val, y_train, y_val = train_test_split(df, target, test_size=val_size, random_state=42)
    X_train.to_csv(data_dir + 'data_train.csv', index=False)
    X_val.to_csv(data_dir + 'data_val.csv', index=False)
    y_train.to_csv(data_dir + 'target_train.csv', index=False)
    y_val.to_csv(data_dir + 'target_val.csv', index=False)
    
    

def main(args):
    split(data_dir=args.data_dir, val_size=args.validation_size)


if __name__ == "__main__":
    args = parse_args()
    main(args)
