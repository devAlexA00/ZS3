import os
import os.path as osp
import pathlib

import numpy as np
import scipy
from PIL import Image
from torchvision import transforms

from zs3.dataloaders import custom_transforms as tr
from .base import BaseDataset, lbl_contains_unseen

# Répertoire des données PlantDoc
PLANTDOC_DIR = pathlib.Path("./data/plantdoc/")


class PlantDocSegmentation(BaseDataset):
    """
    Plantdoc dataset
    """

    NUM_CLASSES = 3

    def __init__(
        self,
        args,
        base_dir=PLANTDOC_DIR,
        split="train",
        load_embedding=None,
        w2c_size=300,
        weak_label=False,
        unseen_classes_idx_weak=2,
        transform=True,
    ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__(
            args,
            base_dir,
            split,
            load_embedding,
            w2c_size,
            weak_label,
            unseen_classes_idx_weak,
            transform,
        )
        # Répertoire des images
        self._image_dir = self._base_dir / "train/images"
        #print("PATH :", self._image_dir)
        # Répertoire des masques
        self._cat_dir = self._base_dir / "train/masks"

        self.unseen_classes_idx_weak = unseen_classes_idx_weak

        _splits_dir = self._base_dir

        self.im_ids = []
        self.categories = []

        lines = (_splits_dir / f"{self.split}.txt").read_text().splitlines()

        # Récupérer les images et les masques
        for ii, line in enumerate(lines):
            _image = self._image_dir / f"{line}.jpg"
            _cat = self._cat_dir / f"{line}.png"
            assert _image.is_file(), _image
            assert _cat.is_file(), _cat

            # if unseen classes and training split
            if self.split == "train":
                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)
                if lbl_contains_unseen(cat, args.unseen_classes_idx):
                    continue

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

        # Display stats
        print(f"(pascal) Number of images in {split}: {len(self.images):d}")

    # Pour les embeddings
    def init_embeddings(self):
        if self.load_embedding == "my_w2c":
            embed_arr = np.load("/Users/alex/Documents/GitHub/ZS3/zs3/embeddings/plantdoc/plantdoc_class_w2c.npy")
        else:
            raise KeyError(self.load_embedding)
        self.make_embeddings(embed_arr)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        if self.weak_label:
            unique_class = np.unique(np.array(_target))
            has_unseen_class = False
            for u_class in unique_class:
                if u_class in self.unseen_classes_idx_weak:
                    has_unseen_class = True
            if has_unseen_class:
                _target = Image.open(
                    "/Users/alex/Documents/GitHub/ZS3/data/plantdoc/train/masks"
                    + self.categories[index].stem
                    + ".jpg"
                )

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split == "train":
                sample = self.transform_tr(sample)
            elif self.split == "val":
                sample = self.transform_val(sample)
        else:
            sample = self.transform_weak(sample)

        if self.load_embedding:
            self.get_embeddings(sample)
        sample["image_name"] = str(self.images[index])
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                    fill=255,
                ),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.FixScale(crop_size=self.args.crop_size),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_weak(self, sample):

        composed_transforms = transforms.Compose(
            [
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def __str__(self):
        return f"PLANTDOC(split={self.split})"
