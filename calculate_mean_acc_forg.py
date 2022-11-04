import os
import glob
import pandas as pd
import numpy as np
import argparse

def calculate_avg_acc(df):
    accs = df.values
    return np.mean(accs[-1,:])

def calculate_avg_forg(df):
    accs = df.values
    max_accs = np.max(accs[:-1, :], axis=1)
    final_accs = accs[-1, :-1]
    
    return np.mean(max_accs-final_accs)
def calculate_avg_acc_multiple_seed(df_path='asset/acc_matrix', args=None):
    full_path = os.path.join(df_path, f"{args.dataset}_mem_{args.memory_size}")
    full_dfs = glob.glob(os.path.join(full_path ,'*.csv'))
    accs, forgs = [], []
    for df_path in full_dfs:
        df = pd.read_csv(df_path, index_col=0)
        accs.append(calculate_avg_acc(df))
        forgs.append(calculate_avg_forg(df))
    print(f"Dataset     : {args.dataset}")
    print(f"Memory Size : {args.memory_size}")
    print(f"Average Accs over Seeds  : {np.mean(np.array(accs)) +- np.std(np.array(accs))}")
    print(f"Average Forgs over Seeds : {np.mean(np.array(forgs)) +- np.std(np.array(forgs))}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_size', type=int, default=200, choices=(200, 500, 2000))
    parser.add_argument('--dataset', type=str, default='cifar100', choices=('cifar10', 'cifar100', 'tinyimagenet'))
    args = parser.parse_args()
    calculate_avg_acc_multiple_seed(args=args)