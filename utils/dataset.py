from typing import Optional, Union, Tuple, List, Dict

import os
from os import path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pickle, h5py


class ModulationFineTuningDataset(Dataset):
    """Base class for modulation datasets."""

    def __init__(
        self,
        features: Union[np.ndarray, torch.FloatTensor],
        labels: Union[np.ndarray, torch.LongTensor],
    ) -> None:
        super().__init__()
        self.features = features
        self.labels = labels.long()

        # The length of the dataset
        self._dataset_length = len(self.labels)

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self._dataset_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a sample and its label by index."""

        feature = self.features[index]
        label = self.labels[index]

        return feature, label


class BaseDataLoader(object):
    """Base class for modulation dataset loaders."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

        self.batch_size = configs.batch_size
        self.num_workers = configs.num_workers
        self.shuffle = configs.shuffle

        # The path of the dataset
        self.root_path = configs.root_path
        self.file_path = configs.file_path

        # The training snrs selected
        self.target_snr = configs.snr

        # Fix the batch size for evaluation and testing
        self.val_batch_size = 128

        # Fix the split ratio
        self.train_split = configs.split_ratio / 3 * 5
        # Get the data for testing
        self.val_test_split_ratio = 0.4

    @classmethod
    def load_pkl(cls, file_path: str) -> Dict:
        """Load a pickle file and return its content as a dictionary."""
        return pickle.load(open(file_path, "rb"), encoding="iso-8859-1")

    @classmethod
    def load_dat(cls, file_path: str) -> Dict:
        """Load a .dat file and return its content as a dictionary."""
        return pickle.load(
            open(file_path, "rb"), encoding="iso-8859-1"
        )  # Xd2(22W,2,128)

    @classmethod
    def load_h5py(cls, file_path: str) -> Dict:
        """Load a .h5py file and return its content as a dictionary."""
        return h5py.File(file_path, "r")

    def get_data_loader(
        self,
        train_dataset: ModulationFineTuningDataset,
        val_dataset: ModulationFineTuningDataset,
        test_dataset: ModulationFineTuningDataset,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
    ) -> Tuple[
        DataLoader,
        DataLoader,
        DataLoader,
    ]:
        """Create data loaders for training, validation, and testing datasets."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size if batch_size else self.batch_size,
            shuffle=shuffle if shuffle is not None else self.shuffle,
            num_workers=self.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader

    def normalization(self, X: np.ndarray) -> np.ndarray:
        """Normalize the dataset to have zero mean and unit variance."""
        # Create the scaler from sklearn
        scaler = StandardScaler()

        # Reshape the data to 2D for scaling
        num_samples, num_channels, signal_length = X.shape
        X_reshaped = X.reshape(-1, signal_length)

        # Fit the scaler and transform the data
        X_scaled = scaler.fit_transform(X_reshaped)
        X_normalized = X_scaled.reshape(num_samples, num_channels, signal_length)

        return X_normalized


class RML2016aDataLoader(BaseDataLoader):
    """Data loader for the RML2016.10a dataset."""

    def __init__(self, configs) -> None:
        super().__init__(configs)

    @property
    def class_list(self) -> List[str]:
        """Return the list of modulation classes in the RML2016.10a dataset."""
        return [
            "8PSK",
            "AM-DSB",
            "AM-SSB",
            "BPSK",
            "CPFSK",
            "GFSK",
            "PAM4",
            "QAM16",
            "QAM64",
            "QPSK",
            "WBFM",
        ]

    @property
    def snr_list(self) -> List[int]:
        """Return the list of SNR values in the RML2016.10a dataset."""
        return [
            -20,
            -18,
            -16,
            -14,
            -12,
            -10,
            -8,
            -6,
            -4,
            -2,
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
        ]

    def load(
        self, batch_size: Optional[int] = None, shuffle: Optional[bool] = None
    ) -> None:
        """Load the RML2016.10a dataset and return data loaders."""
        # Load the dataset
        data_dict = self.load_pkl(file_path=self.file_path)

        mods, snrs = [
            sorted(list(set([k[j] for k in data_dict.keys()]))) for j in [0, 1]
        ]

        # Create data lists for training, validation, and test sets.
        X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for idx, mod in enumerate(mods):
            X = data_dict[(mod, self.target_snr)]
            y = np.ones(X.shape[0]) * idx

            # The proportion of data allocated to the training set is 60%.
            X_train, X_temp, y_train, y_temp = train_test_split(
                X,
                y,
                test_size=self.val_test_split_ratio,
            )

            # Further partition the training dataset.
            if self.configs.split_ratio != 0.6:
                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, train_size=self.train_split
                )

            # The remaining 40% of the data is divided equally into a test set and a validation set.
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5
            )

            # Append the data to the lists
            X_train_list.append(X_train)
            X_val_list.append(X_val)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_val_list.append(y_val)
            y_test_list.append(y_test)

        X_train = np.vstack(X_train_list)
        X_val = np.vstack(X_val_list)
        X_test = np.vstack(X_test_list)
        y_train = np.hstack(y_train_list).astype(int)
        y_val = np.hstack(y_val_list).astype(int)
        y_test = np.hstack(y_test_list).astype(int)

        # Create the dataset objects and do normalization
        train_dataset = ModulationFineTuningDataset(
            features=torch.FloatTensor(self.normalization(X_train)),
            labels=torch.LongTensor(y_train),
        )
        val_dataset = ModulationFineTuningDataset(
            features=torch.FloatTensor(self.normalization(X_val)),
            labels=torch.LongTensor(y_val),
        )
        test_dataset = ModulationFineTuningDataset(
            features=torch.FloatTensor(self.normalization(X_test)),
            labels=torch.LongTensor(y_test),
        )

        # Return the data loaders
        return self.get_data_loader(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )


class RML2016bDataLoader(BaseDataLoader):
    """Data loader for the RML2016.10b dataset."""

    def __init__(self, configs) -> None:
        super().__init__(configs)

    @property
    def class_list(self) -> List[str]:
        """Return the list of modulation classes in the RML2016.10b dataset."""
        return [
            "8PSK",
            "AM-DSB",
            "BPSK",
            "CPFSK",
            "GFSK",
            "PAM4",
            "QAM16",
            "QAM64",
            "QPSK",
            "WBFM",
        ]

    @property
    def snr_list(self) -> List[int]:
        """Return the list of SNR values in the RML2016.10b dataset."""
        return [
            -20,
            -18,
            -16,
            -14,
            -12,
            -10,
            -8,
            -6,
            -4,
            -2,
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
        ]

    def load(
        self, batch_size: Optional[int] = None, shuffle: Optional[bool] = None
    ) -> None:
        """Load the RML2016.10b dataset and return data loaders."""
        # Load the dataset
        data_dict = self.load_dat(file_path=self.file_path)

        mods, snrs = [
            sorted(list(set([k[j] for k in data_dict.keys()]))) for j in [0, 1]
        ]

        # Create data lists for training, validation, and test sets.
        X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # Loop over modulation types
        for idx, mod in enumerate(mods):
            X = data_dict[(mod, self.target_snr)]
            y = np.ones(X.shape[0]) * idx

            # The proportion of data allocated to the training set is 60%.
            X_train, X_temp, y_train, y_temp = train_test_split(
                X,
                y,
                test_size=self.val_test_split_ratio,
            )

            # Further partition the training dataset.
            if self.configs.split_ratio != 0.6:
                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, train_size=self.train_split
                )

            # The remaining 40% of the data is divided equally into a test set and a validation set.
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5
            )

            # Append the data to the respective lists
            X_train_list.append(X_train)
            X_val_list.append(X_val)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_val_list.append(y_val)
            y_test_list.append(y_test)

        # Stack the data from list
        X_train = np.vstack(X_train_list)
        X_val = np.vstack(X_val_list)
        X_test = np.vstack(X_test_list)
        y_train = np.hstack(y_train_list).astype(int)
        y_val = np.hstack(y_val_list).astype(int)
        y_test = np.hstack(y_test_list).astype(int)

        # Create the dataset objects and do normalization
        train_dataset = ModulationFineTuningDataset(
            features=torch.FloatTensor(self.normalization(X_train)),
            labels=torch.LongTensor(y_train),
        )
        val_dataset = ModulationFineTuningDataset(
            features=torch.FloatTensor(self.normalization(X_val)),
            labels=torch.LongTensor(y_val),
        )
        test_dataset = ModulationFineTuningDataset(
            features=torch.FloatTensor(self.normalization(X_test)),
            labels=torch.LongTensor(y_test),
        )

        # Return the data loaders
        return self.get_data_loader(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )


class RML2018aDataLoader(BaseDataLoader):
    """Data loader for the RML2018.10a dataset."""

    def __init__(self, configs) -> None:
        super().__init__(configs)

    @property
    def class_list(self) -> List[str]:
        """Return the list of modulation classes in the RML2016.10b dataset."""
        return [
            "OOK",
            "4ASK",
            "8ASK",
            "BPSK",
            "QPSK",
            "8PSK",
            "16PSK",
            "32PSK",
            "16APSK",
            "32APSK",
            "64APSK",
            "128APSK",
            "16QAM",
            "32QAM",
            "64QAM",
            "128QAM",
            "256QAM",
            "AM-SSB-WC",
            "AM-SSB-SC",
            "AM-DSB-WC",
            "AM-DSB-SC",
            "FM",
            "GMSK",
            "OQPSK",
        ]

    @property
    def snr_list(self) -> List[int]:
        """Return the list of SNR values in the RML2016.10b dataset."""
        return [
            -20,
            -18,
            -16,
            -14,
            -12,
            -10,
            -8,
            -6,
            -4,
            -2,
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
        ]

    def load(
        self, batch_size: Optional[int] = None, shuffle: Optional[bool] = None
    ) -> None:

        data = self.load_h5py(self.file_path)

        X = data["X"]
        y = np.argmax(data["Y"], axis=1)

        # Get the SNR of the data
        idx_snr = np.where(np.array(data["Z"]).flatten() == self.target_snr)[0]

        # Get data under the specified SNR
        X = np.transpose(X[idx_snr], (0, 2, 1))
        y = y[idx_snr]

        # Divide the dataset into training, validation, and test sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.val_test_split_ratio,
        )

        # Further partition the training dataset.
        if self.configs.split_ratio != 0.6:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, train_size=self.train_split
            )

        # Divide the test set and the validation set
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

        # Create the dataset objects and do normalization
        train_dataset = ModulationFineTuningDataset(
            features=torch.FloatTensor(self.normalization(X_train)),
            labels=torch.LongTensor(y_train),
        )
        val_dataset = ModulationFineTuningDataset(
            features=torch.FloatTensor(self.normalization(X_val)),
            labels=torch.LongTensor(y_val),
        )
        test_dataset = ModulationFineTuningDataset(
            features=torch.FloatTensor(self.normalization(X_test)),
            labels=torch.LongTensor(y_test),
        )

        # Return the data loaders
        return self.get_data_loader(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )


class PreTrainingDataLoader(object):
    """Data loader for pre-training datasets."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
