import os
import pickle
from glob import glob

import yaml
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from sklearn.model_selection import train_test_split

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class CardiallyPreparator:

    def __init__(self, target_freq: int=500):

        self.max_length = 9 # sec (from original paper)
        self.tick = 1e5
        self.target_freq = target_freq

        self.save_loc = os.path.join(cfg["settings"]["common"]["save_root"], "cardially")
        os.makedirs(self.save_loc, exist_ok=True)

        self._prepare_array()

    def _prepare_array(self):
        """
        Args:

        Returns:

        """
        targets = self._load_files()
        ecgs = []
        for target in tqdm(targets):
            ecgs.append(self._convert_to_array(target))
        self.ecgs = np.array(ecgs)

    def _load_files(self):
        """
        Args:

        Returns:

        """
        targets = sorted(glob(cfg["settings"]["cardially"]["src"] + "/ROEA/*.txt"))
        targets += sorted(glob(cfg["settings"]["cardially"]["src"] + "/noROEA/*.txt"))

        return targets

    def _convert_to_array(self, target_file: str):
        """
        Args:
            target_file (str): 
        Returns:
            downsampled_ecg (np.ndarray): 
        """

        # Open data.
        txtdata = open(target_file).readlines()

        # Convert to array
        tstamp, val = [], []
        for row in (txtdata):
            row = row.strip().split(" ")
            row = list(filter(lambda x: x != "", row))
            assert len(row) == 2
            tstamp.append(float(row[0]))
            val.append(float(row[1]))
        tstamp = np.array(tstamp)
        val = np.array(val)

        # Fill in observed and interpolate
        ecg = np.zeros(int(self.max_length*self.tick))
        for i in range(len(tstamp)-1):
            start_t = int(tstamp[i] * self.tick)
            end_t = int(tstamp[i+1] * self.tick)
            f = interpolate.interp1d(
                np.array([start_t, end_t]), 
                np.array([val[i], val[i+1]])
            )
            ecg[start_t:end_t] = f(np.arange(start_t, end_t))

        # Downsample.
        ds = int(self.tick / self.target_freq)
        downsampled_ecg = ecg[::ds]

        return downsampled_ecg
    
    def _save_data(self, data: np.ndarray, datatype: str, seed: int=None):
        """
        Args:

        Returns:

        """
        if seed is not None:
            fname = f"{datatype}_seed{seed:04d}.pkl"
        else:
            fname = f"{datatype}.pkl"
        
        savename = os.path.join(
            self.save_loc,
            fname
        )
        
        with open(savename, "wb") as fp:
            pickle.dump(data, fp)

    def make_dataset(self):
        """
        Args:

        Returns:

        """

        Xtr, Xte = train_test_split(
            self.ecgs, 
            test_size=cfg["split"]["test"]["size"], 
            random_state=cfg["split"]["test"]["seed"]
        )
        self._save_data(Xte, "test")

        seeds = cfg["split"]["train_val"]["seeds"]
        for i, seed in enumerate(seeds):
            print(f"{i+1}/{len(seeds)}")
            Xtr_sp, Xv_sp = train_test_split(
                Xtr, 
                test_size=cfg["split"]["train_val"]["size"], 
                random_state=seed
            )
            self._save_data(Xtr_sp, "train", seed)
            self._save_data(Xv_sp, "val", seed)
        print("Done")

if __name__ == "__main__":

    preparator = CardiallyPreparator()
    preparator.make_dataset()
