import yaml
import numpy as np

from gen_ecg import (
    ECGsynthesizer, 
    generate_peak_wave,
    white_noise,
    base_shift
)

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class VFsynthesizer(ECGsynthesizer):

    syn_type = "syn_vf"

    def generate_vf(self, length, params, max_n_beat):
        """
        Args:
        
        Returns:
        
        """
        base_len = self.base_len * (length / self.fs)

        t = np.linspace(
            0, 
            base_len, 
            int(base_len * self.fs), 
            endpoint=False
        )

        wave = np.zeros_like(t)
        shift = params.shift
        for c in range(max_n_beat):
            width = params.width * (np.random.rand() + 0.5) * np.abs(params.width_scale)
            peak = (-1)**c * params.peak + np.random.randn() / np.abs(params.peak_scale)
            wave += generate_peak_wave(t, peak, shift, width)
            shift += np.random.rand() * np.abs(params.shift_scale)
            if shift > base_len:
                break

        # Noise
        wn1 = white_noise(
            wave.shape[0],
            int(params.wn1_width), 
            params.wn1_scaler
        )
        wn2 = white_noise(
            wave.shape[0],
            int(params.wn2_width), 
            params.wn2_scaler
        )

        wave = wave + wn1 + wn2
        start_val = wave[np.random.randint(len(wave))]
        pseudo_ecg = base_shift(wave, start_val, params.base_scale)

        return pseudo_ecg

    def generate_ecg(self):
        """
        Args:
        
        Returns:
        
        """
        p_ecg = np.array([])
        base_params = self.set_base_param()
        max_n_beat = int(100 * (self.target_length * 1.5) / self.fs)

        p_ecg = self.generate_vf(
            self.target_length*1.5, 
            base_params,
            max_n_beat
        )

        # Randomly pick location.
        start_loc = np.random.choice(len(p_ecg) - self.target_length)
        p_ecg = p_ecg[start_loc:start_loc+self.target_length]

        n_aug = np.random.poisson(lam=2.)
        for _ in range(n_aug):
            p_ecg = self.augmentor.rand_augment(p_ecg)

        return p_ecg
    
if __name__ == "__main__":

    for seed in range(1, 7):
        print(f"Working on {seed} ...")
        syn = VFsynthesizer(seed=seed)
        syn.make_dataset()
    print("Done")
