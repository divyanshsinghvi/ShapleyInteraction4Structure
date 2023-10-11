import numpy as np

from transformers import (
    AutoImageProcessor,
    ResNetForImageClassification,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    ViTImageProcessor,
    pipeline,
)
import torch
from datasets import load_dataset

import argparse
import os
import logging

from time import perf_counter
import pickle
import gc

logger = logging.getLogger("dp_logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

from img_utils import *


def main(args):
    assert not (
        args.cifar and args.mnist
    ), "Need to choose one of cifar/mnist for experiments, not both"

    output_dir = os.path.abspath(args.output)

    if not os.path.exists(output_dir):
        logger.info(f"creating output directory {output_dir}")
        os.makedirs(output_dir)

    ref_val = args.reference
    cuda = True if torch.cuda.is_available() else False
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        mps = True
    else:
        mps = False

    split = "test"

    if args.cifar:
        data = load_dataset("cifar100", split=split)
        processor = ViTImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100")
        classifier = AutoModelForImageClassification.from_pretrained(
            "Ahmed9275/Vit-Cifar100"
        )
        img_str = "cifar"

    elif args.mnist:
        data = load_dataset("mnist", split=split)
        processor = ViTImageProcessor.from_pretrained(
            "farleyknight-org-username/vit-base-mnist"
        )
        classifier = AutoModelForImageClassification.from_pretrained(
            "farleyknight-org-username/vit-base-mnist"
        )
        img_str = "mnist"

    if cuda:
        classifier.to("cuda")
    elif mps:
        classifier.to("mps")

    img_processor = ImageProcessor(
        processor=processor,
        classifier=classifier,
        reference_value=ref_val,
        cuda=cuda,
        phi=args.phi,
        data_id=img_str,
    )

    if img_str == "mnist":
        images = [
            np.stack([np.array(data[idx]["image"])] * 3, axis=-1)
            for idx in range(args.num_samples)
        ]
    else:
        images = [np.array(data[idx]["image"]) for idx in range(args.num_samples)]

    dataset = CombDataset(images)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    inf_values = img_processor.run_inference(dataloader=dataloader)
    start = perf_counter()
    for idx, img in enumerate(images):
        inter = img_processor.get_interactions(img, idx, inf_values=inf_values)
        path = os.path.join(output_dir, f"interactions_{img_str}_{split}_{idx}.pickle")
        with open(path, "wb") as f:
            pickle.dump(inter, f)
        logger.info(
            f"interactions for image {idx} saved to {path}.\nGenerations took {perf_counter() - start} seconds"
        )
        start = perf_counter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Output path")
    parser.add_argument(
        "-p",
        "--phi",
        action="store_true",
        help="Whether to add phi into the calculation",
    )
    parser.add_argument(
        "-c", "--cifar", action="store_true", help="Use cifar dataset if passed"
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        help="Num samples to calculate values for on dataset.",
    )
    parser.add_argument(
        "-m", "--mnist", action="store_true", help="Use mnist dataset if passed"
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=int,
        help="Reference value for image, if 0 then pixel values will be set to 0, else noise will be added",
    )

    args = parser.parse_args()
    main(args)
