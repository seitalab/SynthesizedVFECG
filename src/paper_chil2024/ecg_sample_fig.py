import os
import sys
import pickle

import numpy as np

sys.path.append("..")
from utils.ecg_plot import make_ecg_plot

root = "../../dataset/v240205"
duration = 10
freq = 500
save_root = "result_figs"

def load(dataset, datatype, seed):

    path = os.path.join(
        root,
        dataset,
        f"{datatype}_seed{seed:04d}.pkl"
    )
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data

def make_fig(dataset, datatype, seed, n_fig):
    
    np.random.seed(seed)
    data = load(dataset, datatype, seed)
    
    target_idxs = np.random.choice(
        np.arange(len(data)), size=n_fig, replace=False)
    
    save_dir = os.path.join(
        save_root,
        f"samples-{dataset}"
    )
    os.makedirs(save_dir, exist_ok=True)

    for target_idx in target_idxs:
        savename = os.path.join(
            save_dir,
            f"idx{target_idx:06d}.png"
        )
        make_ecg_plot(
            data[target_idx], duration, freq, savename)
    print("done")

if __name__ == "__main__":

    dataset = sys.argv[1]
    seed = 1
    datatype = "train"
    n_fig = 3    
    make_fig(dataset, datatype, seed, n_fig)
