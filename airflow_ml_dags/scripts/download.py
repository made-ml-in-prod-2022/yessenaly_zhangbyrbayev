import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-od", "--output_dir", 
                        dest="output_dir", 
                        type=str, 
                        default=None, 
                        help="directory where we save downloaded data"
                        )
    return parser.parse_args()

def save(df: pd.DataFrame, path: str):
    if not os.path.exists(path):
        df.to_csv(path, 
                    index=False
                )
    else:
        df.to_csv(path, 
                    mode='a', 
                    header=not os.path.exists(path), 
                    index=False
                )

def download(output_dir: str):
    X, y = load_iris(return_X_y=True, as_frame=True)
    idx = np.random.randint(X.shape[0], size=5)
    X_append, y_append = X.loc[idx], y.loc[idx]
    save(df=X_append, path=output_dir + 'data.csv')
    save(df=y_append, path=output_dir + 'target.csv')

def main(args):
    download(output_dir=args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
