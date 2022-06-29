import argparse
import json
from argparse import Namespace
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]


def load_params(path: Path) -> Namespace:
    """Load a parameters file. The file is obtained from the model's wandb run.

    Parameters
    ----------
    path : Path
        Path to params.json file.

    Returns
    -------
    Namespace
        Loaded parameters.
    """
    with open(path, "r") as f:
        params = json.load(f)
    params = {arg: val for arg, values in params.items() for key, val in values.items()}
    return Namespace(**params)


def join_namespaces(args1: Namespace, args2: Namespace) -> Namespace:
    """Join two Namespaces into the first one.

    Parameters
    ----------
    args1 : Namespace
        First Namespace.
    args2 : Namespace
        Second Namespace.

    Returns
    -------
    Namespace
        First Namespace updated with arguments from Namespace 2.
    """
    args1 = vars(args1)
    args1.update(vars(args2))
    return Namespace(**args1)


def main(args):
    print(f"--- Generating samples for {args.model}: {args.weights_path}")
    if args.model == "original_data":
        if args.invert_p is None:
            raise RuntimeError("Please provide the `invert_p` argument.")

        from src.data.generate_samples_original import generate_samples

        generate_samples(
            args.invert_p,
            args.batch_size,
            args.num_samples,
            mnist_folder=SRC_DIR / "data/raw",
            output_dir=args.output_dir,
        )
    else:
        if args.weights_path is None or args.params_path is None:
            raise ValueError(
                "Please provide the arguments `weights_path` and `params_path`."
            )

        if args.model == "ddgan":
            from src.data.generate_samples_ddgan import generate_samples

        elif args.model == "stylegan2":
            raise NotImplementedError("StyleGAN2 sample generation is not implemented.")

        elif args.model == "wgan":
            from src.data.generate_samples_WGAN import generate_samples

        generate_samples(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate samples for suppoted experiments")
    parser.add_argument(
        "model",
        type=str,
        help="model to generate samples from",
        choices=["original_data", "ddgan", "stylegan2", "wgan"],
    )
    parser.add_argument(
        "--weights_path", type=Path, default=None, help="path to weights file"
    )
    parser.add_argument(
        "--params_path", type=Path, default=None, help="path to params.json file"
    )

    parser.add_argument(
        "--invert_p",
        type=float,
        default=None,
        choices=[None, 0.3, 0.5, 0.7],
        help="probability to invert image colors when `model == original_data`",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="number of images to be generated",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="generation batch size"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SRC_DIR / "data/generated_samples",
        help="output folder",
    )
    args = parser.parse_args()

    if args.params_path:
        params = load_params(args.params_path)
        args = join_namespaces(params, args)

    main(args)
