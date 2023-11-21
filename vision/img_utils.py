import numpy as np
import torch
import cv2
from copy import deepcopy
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
from time import perf_counter

from tqdm import tqdm


class ImageProcessor:
    def __init__(self, processor, classifier, reference_value, cuda, phi, data_id):
        self.processor = processor
        self.classifier = classifier
        self.cuda = cuda
        self.phi = phi
        self.data_id = data_id

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.mps = True
        else:
            self.mps = False

        self.ref_value = reference_value

    @staticmethod
    def get_all_pairs(img: np.array) -> List[Tuple]:
        print(img.shape)
        if len(img.shape) == 2:
            height, width = img.shape
        else:
            height, width, _ = img.shape
        pairs = []
        for i in range(width):
            for j in range(height):
                for k in range(i, width):
                    for l in range(height):
                        if k == i and l <= j:
                            continue
                        pairs.append(((j, i), (l, k)))

        return pairs

    def get_shap_val(self, p1, p2, img, instruction):
        img1 = np.copy(img)

        if instruction == "a+b":
            pass
        elif instruction == "a":
            img1[p1] = self.ref_value
        elif instruction == "b":
            img1[p2] = self.ref_value
        elif instruction == "phi":
            img1[p1] = self.ref_value
            img1[p2] = self.ref_value

        inputs = self.processor(img, return_tensors="pt")

        if self.cuda:
            inputs.to("cuda")

        with torch.no_grad():
            logits = self.classifier(**inputs).logits

        return logits.softmax(dim=-1)

    def run_inference(self, dataloader):
        d = {}
        batch = 0
        with torch.no_grad():
            for batch_images, info in tqdm(dataloader, desc="Inference batches"):
                if self.cuda:
                    processed = self.processor(batch_images, return_tensors="pt").to(
                        "cuda"
                    )
                elif self.mps:
                    processed = self.processor(batch_images, return_tensors="pt").to(
                        "mps"
                    )
                outputs = self.classifier(**processed).logits
                soft_out = outputs.softmax(dim=-1).cpu()
                update = dict(zip(info, soft_out))
                d.update(update)
                batch += 1

        return d

    def get_interaction(self, pixel_pair, img):
        p1, p2 = pixel_pair

        a = self.get_shap_val(p1, p2, img, "a")
        b = self.get_shap_val(p1, p2, img, "b")
        apb = self.get_shap_val(p1, p2, img, "a+b")

        if self.phi:
            phi = self.get_shap_val(p1, p2, img, "phi")
            num = apb - a - b + phi
        else:
            num = apb - a - b

        if self.cuda:
            num = torch.linalg.norm(num, dim=-1).cpu()
            den = torch.linalg.norm(apb, dim=-1).cpu()
        else:
            num = np.linalg.norm(num, axis=-1)
            den = np.linalg.norm(apb, axis=-1)

        return num / den

    def get_interactions(self, img, idx, inf_values: dict):
        all_pairs = self.get_all_pairs(img)
        interactions = {}
        for (x1, y1), (x2, y2) in all_pairs:
            a = inf_values[f"image{idx}_zero_{x1}_{y1}"]
            b = inf_values[f"image{idx}_zero_{x2}_{y2}"]
            apb = inf_values[f"image{idx}_original"]
            if self.phi:
                phi = inf_values[f"image{idx}_zero_{x1}_{y1}_{x2}_{y2}"]
                num = apb - a - b + phi
            else:
                num = apb - a - b
            num = np.linalg.norm(num, ord=2, axis=-1)
            den = np.linalg.norm(apb, ord=2, axis=-1)

            int_val = num / den
            update = {((x1, y1), (x2, y2)): int_val}
            interactions.update(update)

        return interactions


class CombDataset(Dataset):
    def __init__(self, images: list, reference_value: str):
        """
        Dataset for data loader for inference.

        If reference value is set to 'blur', currently we choose a 5x5 kernel size.

        Parameters:
        -----------
        images: List[np.array]
            numpy version of images

        reference_value: str
            reference value to choose to set interaction pixels to, must be one of ["zero", "mean", "blur"]

        """
        self.images = images
        self.H, self.W = images[0].shape[:2]
        self.pairs = ImageProcessor.get_all_pairs(images[0])
        if reference_value == 0:
            self.ref_type = "zero"
        elif reference_value == 1:
            self.ref_type = "mean"
        elif reference_value == 2:
            self.ref_type = "blur"
        else:
            raise ValueError(f"{reference_value} not a valid reference value")

        print(f"REFERENCE VALUE: {self.ref_type}")
        self.is_twod = True if len(self.images[0].shape) == 2 else False

    def get_reference_value(self, img):
        if self.ref_type == "zero":
            refs = np.zeros(img.shape)
        elif self.ref_type == "mean":
            if self.is_twod:
                mean = np.mean(img)
                refs = np.repeat(img.shape, mean)
            else:
                refs = np.zeros(img.shape)
                mean = np.mean(img, axis=(0, 1))
                for channel in range(len(mean)):
                    refs[..., channel] = mean[channel]
        elif self.ref_type == "blur":
            refs = cv2.blur(img, (3, 3))
        else:
            raise ValueError(
                "Reference value type not supported, must be one of ['zero', 'mean', 'blur']"
            )

        return refs

    def __len__(self):
        # Original + every pixel + every pair of pixels
        return len(self.images) * (1 + (self.H * self.W) + (len(self.pairs)))

    def __getitem__(self, idx):
        H, W = self.H, self.W
        # Place holder for image indexing
        image_idx = idx // (1 + H * W + len(self.pairs))
        inner_idx = idx % (1 + (H * W) + len(self.pairs))

        img = np.copy(self.images[image_idx])
        img_refs = self.get_reference_value(self.images[image_idx])

        if inner_idx == 0:
            return img, f"image{image_idx}_original"

        elif inner_idx <= H * W:
            x, y = (inner_idx - 1) // W, (inner_idx - 1) % W
            img[x, y, :] = img_refs[x, y, :]
            return img, f"image{image_idx}_zero_{x}_{y}"

        else:
            i, j = self.pairs[inner_idx - (H * W) - 1]
            x1, y1 = i
            x2, y2 = j
            img[x1, y1, :] = img_refs[x1, y1, :]
            img[x2, y2, :] = img_refs[x2, y2, :]
            return img, f"image{image_idx}_zero_{x1}_{y1}_{x2}_{y2}"
