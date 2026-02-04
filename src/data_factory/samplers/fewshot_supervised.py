import random
from torch.utils.data import Sampler
from ..dataset_task.Dataset_cluster import IdIncludedDataset


class FewShotSupervisedSampler(Sampler):
    """Few-shot supervised sampler by label (window-level).

    Each episode samples N-way * K-shot window indices for training.
    """

    def __init__(
        self,
        dataset: IdIncludedDataset,
        n_way: int,
        k_shot: int,
        episodes_per_epoch: int = 1,
        label_key: str = "Label",
    ):
        if not isinstance(dataset, IdIncludedDataset):
            raise ValueError("dataset must be an IdIncludedDataset.")
        if not hasattr(dataset, "metadata") or dataset.metadata is None:
            raise ValueError("dataset must have metadata.")
        if n_way <= 0 or k_shot <= 0:
            raise ValueError("n_way and k_shot must be positive.")

        self.dataset = dataset
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.episodes_per_epoch = max(int(episodes_per_epoch), 1)
        self.label_key = label_key

        self.label_to_indices = {}
        for global_idx, sample_info in enumerate(self.dataset.file_windows_list):
            file_id = sample_info["file_id"]
            meta_entry = self.dataset.metadata.get(file_id, None)
            if meta_entry is None or self.label_key not in meta_entry:
                continue
            label = meta_entry[self.label_key]
            self.label_to_indices.setdefault(label, []).append(global_idx)

        self.labels = list(self.label_to_indices.keys())
        if not self.labels:
            print("[FewShotSupervisedSampler] Warning: no labels found; sampler is empty.")

        self._effective_n_way = min(self.n_way, len(self.labels))
        self._num_samples_epoch = self.episodes_per_epoch * self._effective_n_way * self.k_shot
        self._warned_labels = set()

    def __iter__(self):
        if not self.labels:
            return iter([])

        for _ in range(self.episodes_per_epoch):
            if len(self.labels) < self.n_way:
                if "n_way" not in self._warned_labels:
                    print(
                        "[FewShotSupervisedSampler] Warning: "
                        f"labels({len(self.labels)}) < n_way({self.n_way}); "
                        "using all labels."
                    )
                    self._warned_labels.add("n_way")
                chosen_labels = list(self.labels)
            else:
                chosen_labels = random.sample(self.labels, self.n_way)

            batch_indices = []
            for label in chosen_labels:
                indices = self.label_to_indices.get(label, [])
                if not indices:
                    continue
                if len(indices) >= self.k_shot:
                    selected = random.sample(indices, self.k_shot)
                else:
                    if label not in self._warned_labels:
                        print(
                            "[FewShotSupervisedSampler] Warning: "
                            f"label {label} has {len(indices)} samples < "
                            f"k_shot({self.k_shot}); sampling with replacement."
                        )
                        self._warned_labels.add(label)
                    selected = random.choices(indices, k=self.k_shot)
                batch_indices.extend(selected)

            yield batch_indices

    def __len__(self):
        return self.episodes_per_epoch

    @property
    def num_samples_epoch(self):
        return self._num_samples_epoch
