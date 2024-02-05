import numpy as np
import torch

np.random.seed(0)

class RandomMask:

    def __init__(self, mask_ratio: float):

        self.mask_ratio = mask_ratio

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        mask_width = int(data.shape[0] * self.mask_ratio)
        mask_start = np.random.choice(data.shape[0] - mask_width, 1)[0]

        masked_data = data.copy()
        masked_data[mask_start:mask_start+mask_width] = 0

        return {"data": masked_data}

class RandomShift:

    def __init__(self, max_shift_ratio: float):

        self.max_shift_ratio = max_shift_ratio

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        shift_ratio = np.random.rand() * self.max_shift_ratio
        shift_size = int(data.shape[0] * shift_ratio)

        pad = np.zeros(shift_size)

        shifted_data = data.copy()
        if np.random.rand() < 0.5:
            shifted_data = np.concatenate([pad, shifted_data])[:len(data)]
        else:
            shifted_data = np.concatenate([shifted_data, pad])[-len(data):]
        assert len(data) == len(shifted_data)
        return {"data": shifted_data}

class AlignLength:

    def __init__(self, target_len: int):

        self.target_len = target_len

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]

        if len(data) < self.target_len:
            total_pad = self.target_len - len(data)
            pad_l = int(np.random.rand() * total_pad)
            pad_r = total_pad - pad_l
            data = np.concatenate([
                np.zeros(pad_l),
                data,
                np.zeros(pad_r)
            ])
        
        if len(data) > self.target_len:
            total_cut = len(data) - self.target_len
            cut_l = int(np.random.rand() * total_cut)
            data = data[cut_l:cut_l+self.target_len]

        return {"data": data}

class ScaleECG:

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]

        data = (data - data.mean()) / data.std()

        # data = data[::10]

        return {"data": data}

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        # if self.load_masked:
        #     data, masked_data = sample["data"], sample["masked"]
        #     data_tensor = torch.from_numpy(data)
        #     masked_tensor = torch.from_numpy(masked_data).unsqueeze(0)
        #     sample = {"data": data_tensor, "masked": masked_tensor}
        # else:
        data = sample["data"]
        data_tensor = torch.from_numpy(data)
        sample = {"data": data_tensor}
        return sample
