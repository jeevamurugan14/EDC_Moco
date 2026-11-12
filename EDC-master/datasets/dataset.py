import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def divide255(image, **kwargs):
    return (image / 255.0).astype("float32")


def get_transform(img_size=256, crop_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CenterCrop(crop_size, crop_size),
    ])


class BasicDataset(Dataset):
    def __init__(self, img_paths, targets=None, transform=None, imagenet_norm=True):
        self.img_paths = img_paths
        self.targets   = targets
        self.transform = transform
        self.totensor  = A.Compose([
            A.Normalize() if imagenet_norm else A.Lambda(image=divide255),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        target   = self.targets[idx] if self.targets is not None else 0
        img      = np.array(pil_loader(self.img_paths[idx]))
        img_path = self.img_paths[idx]

        if self.transform is not None:
            img_t = self.transform(image=img)["image"]
        else:
            img_t = img

        img_n = self.totensor(image=img_t)["image"]
        return idx, img_n, img_t, target, img_path


class AD_Dataset_WithVal:
    def __init__(self, name="lungct", img_size=256, crop_size=256,
                 train=True, data_dir="../data",
                 val_fraction=0.2, seed=42,
                 train_samples_limit=10000, imagenet_norm=True):
        self.name        = name
        self.train       = train
        self.data_dir    = data_dir
        self.val_fraction = val_fraction
        self.seed        = seed
        self.train_samples_limit = train_samples_limit
        self.imagenet_norm = imagenet_norm
        self.transform   = get_transform(img_size, crop_size)
        self._split_done = False

    def _get_normal_train_paths(self):
        train_path = os.path.join(self.data_dir, "train", "NORMAL")
        files = [f for f in os.listdir(train_path) if not f.startswith(".")]
        if len(files) > self.train_samples_limit:
            rng = random.Random(self.seed)
            files = rng.choices(files, k=self.train_samples_limit)
        return [os.path.join(train_path, f) for f in files]

    def _get_test_paths_and_labels(self):
        img_paths, targets = [], []
        test_root = os.path.join(self.data_dir, "test")
        for sub_dir in os.listdir(test_root):
            if sub_dir.startswith("."):
                continue
            sub_path = os.path.join(test_root, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            files = [f for f in os.listdir(sub_path) if not f.startswith(".")]
            paths = [os.path.join(sub_path, f) for f in files]
            img_paths.extend(paths)
            label = 0 if sub_dir.upper() == "NORMAL" else 1
            targets.extend([label] * len(paths))
        return img_paths, targets

    def _make_split(self):
        all_paths = self._get_normal_train_paths()
        rng = random.Random(self.seed)
        rng.shuffle(all_paths)
        n_val   = max(1, int(len(all_paths) * self.val_fraction))
        n_train = len(all_paths) - n_val
        self._train_paths = all_paths[:n_train]
        self._val_paths   = all_paths[n_train:]
        print(f"[Dataset Split] Normal train: {n_train}  |  Normal val: {n_val}")
        self._split_done  = True

    def get_train_dset(self):
        if not self._split_done:
            self._make_split()
        targets = np.zeros(len(self._train_paths))
        return BasicDataset(self._train_paths, targets,
                            transform=self.transform,
                            imagenet_norm=self.imagenet_norm)

    def get_val_dset(self):
        if not self._split_done:
            self._make_split()
        targets = np.zeros(len(self._val_paths))
        return BasicDataset(self._val_paths, targets,
                            transform=self.transform,
                            imagenet_norm=self.imagenet_norm)

    def get_eval_dset(self):
        img_paths, targets = self._get_test_paths_and_labels()
        targets = np.array(targets)
        print(f"[Eval Set] Normal: {(targets==0).sum()}  "
              f"Abnormal: {(targets==1).sum()}  Total: {len(targets)}")
        return BasicDataset(img_paths, targets,
                            transform=self.transform,
                            imagenet_norm=self.imagenet_norm)

    def get_dset(self):
        if self.train:
            return self.get_train_dset()
        else:
            return self.get_eval_dset()
