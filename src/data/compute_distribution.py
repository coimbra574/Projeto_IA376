import re
from pathlib import Path

import fire
import pandas as pd
import seaborn as sns
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


def main(
    folder: str, output_dir: str, batch_size: int = 50, threshold: float = 0.5
) -> None:
    """Generate the density plots from mnist digits.

    Parameters
    ----------
    folder : str
        Folder with generated samples separated by `model_proportion` folder.
    output_dir : str
        Folder to save the density figure and results csv to.
    batch_size : int, optional
        Processing batch size, by default 50
    threshold : float, optional
        Threshold to which the image's pixels will be compared to, by default 0.5

    Raises
    ------
    RuntimeError
        Raised when any folder within `folder` don't match the
        regex pattern `([a-zA-Z_]*_[0-9.0-9])`.

    """
    folder = Path(folder)
    output_dir = Path(output_dir)
    csv_path = output_dir / "results.csv"
    img_path = output_dir / "distributions.png"

    if not (csv_path).exists():
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(),])

        dataset = torchvision.datasets.ImageFolder(folder, transform=transform)
        classes = dataset.classes

        if any(
            [re.match(r"([a-zA-Z0-9]*([a-z_A-Z]*)?_[0-9]\.[0-9])", _class) is None for _class in classes]
        ):
            raise RuntimeError(
                f"All folders within {folder} should match the pattern: "
                "name_proportion. For example: ddgan_0.3"
            )

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        results = pd.DataFrame()
        for x, y in tqdm(data_loader):
            avg_pixel = x.mean(axis=-1).mean(axis=-1).reshape(-1)
            higher_than_thresh = (avg_pixel >= threshold).float()
            y_str = list(map(lambda i: classes[i], y.detach().numpy()))
            batch_results = pd.DataFrame(
                {
                    "x_avg_pixel": avg_pixel,
                    "proportion": [_class.split("_")[-1] for _class in y_str],
                    "higher_than_threshold": higher_than_thresh,
                    "model": [_class.split("_")[0] for _class in y_str],
                }
            )
            results = pd.concat([results, batch_results], axis=0)
        results = results.reset_index(drop=True)
    else:
        results = pd.read_csv(csv_path)

    num_samples = results.groupby(["model", "proportion"])["x_avg_pixel"].count()
    print(num_samples)

    plt.figure()
    g = sns.displot(
        data=results, hue="model", x="x_avg_pixel", kind="kde", col="proportion",
    )
    g.set_xlabels("Average pixel value")
    g.fig.subplots_adjust(top=0.8)
    g.fig.suptitle(
        f"Density plots of {num_samples.unique().max()} samples of each model and proportion"
    )
    plt.savefig(img_path, dpi=300)

    if not csv_path.exists():
        results.to_csv(csv_path)


if __name__ == "__main__":
    fire.Fire(main)
