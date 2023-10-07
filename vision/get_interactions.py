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

    img_processor = ImageProcessor(
        processor=processor,
        classifier=classifier,
        reference_value=ref_val,
        cuda=cuda,
        phi=args.phi,
        data_id=img_str,
    )

    for idx in range(args.num_samples):
        interactions = []
        if img_str == "mnist":
            img = np.array(data[idx]["image"])

            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            img = img.transpose((2, 0, 1))
        else:
            img = np.array(data[idx]["img"])
        all_pairs = img_processor.get_all_pairs(img)
        logger.info(f"all pairs for image {idx} generated")

        for p, pair in enumerate(all_pairs):
            interactions.append(img_processor.get_interaction(pair, img))

        logger.info(f"all interactions for iamge {idx} calculated")
        stacked = torch.cat(interactions)
        path = os.path.join(output_dir, f"interactions_{img_str}_{split}_{idx}.pt")
        torch.save(stacked, path)
        logger.info(f"interactions for image {idx} saved to {path}")


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
