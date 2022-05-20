from collections import Counter
from typing import Dict, List

import numpy as np
import torchvision.datasets as datasets
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


class ControlledDataset:
    """Generate a dataset with controlled proportion of class instances.
    """

    def __init__(
        self,
        classes: List,
        proportion: List[float],
        base_dataset: str = "MNIST",
        random_state: int = 42,
    ) -> None:
        self.classes = classes
        self.proportion = proportion
        self.random_state = random_state
        if base_dataset == "MNIST":
            self._base_dataset = datasets.MNIST(
                download=True, root="./data", transform=transforms.ToTensor()
            )
        else:
            raise NotImplementedError("Invalid dataset.")

        self.check_arguments()

        self.sampling_strategy = self.build_sampling_strategy(classes, proportion)

    def build_sampling_strategy(
        self, classes: List, proportion: List
    ) -> Dict[str, int]:
        """Build `sampling_strategy` dictionary according to expected by imblearn.

        Imbalanced learn library expects a dictionary with the amount of instances
        of each class to perform the undersampling. This function builds that dictionary
        from given classes and proportion lists.

        Parameters
        ----------
        classes : List
            List of classes.
        proportion : List
            Proportion of each class.

        Returns
        -------
        Dict[str, int]
            A dictionary with the amount of instances of each class.

        """
        targets = self._base_dataset.targets.numpy()
        unique_classes = np.unique(targets)
        class_count = Counter(targets)

        # Get index, prop and class of maximum desired proportion
        max_idx = np.argmax(proportion)
        max_proportion = proportion[max_idx]
        max_class = classes[max_idx]

        # Get the amount of instances of class with maximum desired proportion
        max_class_qty = class_count[max_class]

        sampling_strategy = {}

        def compute_qty(
            max_proportion: float, max_qty: int, curr_proportion: float
        ) -> int:
            """Compute the number of instances of a given class and desired proportion."""
            return np.floor((max_qty * curr_proportion) / max_proportion).astype(int)

        for c, curr_proportion in zip(classes, proportion):
            class_total_qty = class_count[c]
            class_prop_qty = compute_qty(max_proportion, max_class_qty, curr_proportion)

            # Ensure we have enough instances
            # TODO what happens if we don't?
            assert class_prop_qty <= class_total_qty

            sampling_strategy[c] = class_prop_qty

        # Fill other classes with 0
        sampling_strategy = {
            c: 0 if c not in classes else sampling_strategy[c] for c in unique_classes
        }

        return sampling_strategy

    def check_arguments(self) -> None:
        """Validate main class arguments."""
        if any([c not in self._base_dataset.targets for c in self.classes]):
            raise ValueError("Invalid classes list.")
        if not sum(self.proportion) == 1.0:
            raise ValueError("`proportion` should sum to 1.")

    def __call__(self) -> Subset:
        """Generate a subset of base dataset respecting desired classes and proportions.

        Returns
        -------
        Subset
            A Subset of the original dataset with controlled classes and proportions.

        """
        undersampler = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy, random_state=self.random_state
        )
        targets = self._base_dataset.targets.numpy()
        indexes = np.arange(len(targets)).reshape(-1, 1)
        idx_to_keep, y = undersampler.fit_resample(indexes, targets)
        idx_to_keep = np.sort(idx_to_keep.reshape(-1))

        assert idx_to_keep.shape[0] == sum(self.sampling_strategy.values()), (
            "The number of selected instances does not match the "
            "sampling strategy dictionary."
        )
        assert all(
            c in y for c in self.classes
        ), "Final target array contains undesired classes."

        return Subset(self._base_dataset, idx_to_keep)


if __name__ == "__main__":

    ds = ControlledDataset(
        classes=[0, 1], proportion=[0.8, 0.2], base_dataset="MNIST"
    )()
    dataloader = DataLoader(ds, batch_size=10)

    sample = next(iter(dataloader))
    print("x shape: ", sample[0].shape)
    print("y: ", sample[1])
