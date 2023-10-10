import numpy as np
import torch
from copy import deepcopy
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
from time import perf_counter


class ImageProcessor:
    def __init__(self, processor, classifier, reference_value, cuda, phi, data_id):
        self.processor = processor
        self.classifier = classifier
        self.ref_value = reference_value
        self.cuda = cuda
        self.phi = phi
        self.data_id = data_id

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
        batch_size = 64
        batch = 0
        start = perf_counter()
        with torch.no_grad():
            for batch_images, info in dataloader:
                processed = self.processor(batch_images, return_tensors="pt")
                outputs = self.classifier(**processed).logits
                soft_out = outputs.softmax(dim=-1).cpu()
                update = dict(zip(info, soft_out))
                d.update(update)
                batch += 1
                if batch % 10 == 0:
                    print(
                        f"Time for last {batch_size * batch} samples: {perf_counter()-start}"
                    )
                    start = perf_counter()
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

    def get_interactions(self, img, inf_values: dict):
        all_pairs = self.get_all_pairs(img)
        interactions = {}
        for (x1, y1), (x2, y2) in all_pairs:
            a = inf_values[f"zero_{x1}_{y1}"]
            b = inf_values[f"zero_{x2}_{y2}"]
            apb = inf_values["original"]
            if self.phi:
                phi = inf_values[f"zero_{x1}_{y1}_{x2}_{y2}"]
                num = apb - a - b + phi
            else:
                num = apb - a - b
            num = np.linalg.norm(num, ord=2, axis=-1)
            den = np.linalg.norm(apb, ord=2, axis=-1)

            int_val = num / den
            update = {all_pairs: int_val}
            interactions.update(update)

        return interactions


class CombDataset(Dataset):
    def __init__(self, image):
        self.image = image
        self.H, self.W = image.shape[:-1]

    def __len__(self):
        # Original + every pixel + every pair of pixels
        return 1 + self.H * self.W + (self.H * self.W * (self.H * self.W - 1)) // 2

    def __getitem__(self, idx):
        n_images = len(self.image)
        H, W = self.H, self.W
        image_idx = idx // (1 + H * W + (H * W * (H * W - 1)) // 2)
        inner_idx = idx % (1 + H * W + (H * W * (H * W - 1)) // 2)

        img = np.copy(self.image)

        if inner_idx == 0:
            return img, "original"

        elif inner_idx <= H * W:
            x, y = (inner_idx - 1) // W, (inner_idx - 1) % W
            img[x, y, :] = 0
            return img, f"zero_{x}_{y}"

        else:
            inner_idx -= H * W
            pairs = [(i, j) for i in range(H * W) for j in range(i + 1, H * W)]
            i, j = pairs[inner_idx]
            x1, y1 = i // W, i % W
            x2, y2 = j // W, j % W
            img[x1, y1, :] = 0
            img[x2, y2, :] = 0
            return img, f"zero_{x1}_{y1}_{x2}_{y2}"
