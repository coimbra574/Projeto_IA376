import warnings
from pathlib import Path

import fire
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm


def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataset(invert_p: float, mnist_folder="data") -> MNIST:
    """Generate MNIST dataset.

    Parameters
    ----------
    invert_p : float
        Proportion of samples to have their color inverted.
    mnist_folder : Path
        Where the MNIST dataset is stored, by default "data"

    Returns
    -------
    MNIST
        MNIST dataset
    """
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomInvert(p=invert_p),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = MNIST(mnist_folder, train=True, transform=transform, download=True)

    return dataset


def generate_samples(
    invert_p: float,
    batch_size: int,
    num_samples: int,
    mnist_folder: Path,
    output_dir: Path,
) -> None:
    """Generate MNIST samples with randomized color invert.

    Parameters
    ----------
    invert_p : float
        Proportion of samples to have their color inverted.
    batch_size : int
        Sampling batch size.
    num_samples : int
        Number of total images to generate.
    mnist_folder : Path
        Where the MNIST dataset is stored.
    output_dir : Path
        Output folder to save the samples.
    """

    set_seed(1024)

    dataset = get_dataset(invert_p, mnist_folder)
    size = len(dataset)
    if num_samples is not None and num_samples > size:
        warnings.warn(
            f"`num_samples` is higher than the dataset size: {size}. "
            f"Only {size} images will be saved."
        )

    g = torch.Generator()
    g.manual_seed(1024)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        drop_last=True,
    )

    save_dir = output_dir / f"original_{invert_p}"
    save_dir.mkdir(exist_ok=True, parents=True)
    save_dir = str(save_dir)
    print(f"Saving in {save_dir}")

    with tqdm(total=num_samples) as pbar:
        for i, (images, _) in enumerate(data_loader):
            for j, x in enumerate(images):
                index = i * batch_size + j
                torchvision.utils.save_image(
                    x, "{}/{}.jpg".format(save_dir, str(index).zfill(5))
                )
                if index >= (num_samples - 1):
                    return
            pbar.update(batch_size)


if __name__ == "__main__":
    fire.Fire(generate_samples)
