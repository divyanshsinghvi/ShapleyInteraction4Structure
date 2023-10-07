import numpy as np
import torch
from copy import deepcopy
from typing import List, Tuple


class ImageProcessor:
    def __init__(self, processor, classifier, reference_value, cuda, phi, data_id):
        self.processor = processor
        self.classifier = classifier
        self.ref_value = reference_value
        self.cuda = cuda
        self.phi = phi
        self.data_id = data_id

    def get_all_pairs(self, img: np.array) -> List[Tuple]:
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
