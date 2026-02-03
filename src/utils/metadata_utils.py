"""Metadata helpers shared across models/tasks."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch


def resolve_batch_metadata(
    metadata: Any,
    file_id_batch: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve dataset ids and sample rates for a batch of file_ids.

    Returns:
        dataset_ids: LongTensor[B]
        sample_rates: FloatTensor[B]
    """
    if isinstance(file_id_batch, (list, tuple)):
        rows = [metadata[fid] for fid in file_id_batch]
    elif isinstance(file_id_batch, torch.Tensor):
        rows = [metadata[fid.item()] for fid in file_id_batch.view(-1)]
    else:
        rows = [metadata[file_id_batch]]

    dataset_ids: List[int] = []
    sample_rates: List[float] = []

    for row in rows:
        ds = row["Dataset_id"]
        if isinstance(row, dict):
            fs = row.get("Sample_rate")
            if fs is None:
                fs = row.get("Sample_Rate", row.get("sample_rate", row.get("sample_rate_hz")))
            if fs is None:
                raise KeyError("Sample_rate")
        else:
            fs = row.get("Sample_rate")
            if fs is None:
                fs = row.get("Sample_Rate", row.get("sample_rate", row.get("sample_rate_hz")))
            if fs is None:
                raise KeyError("Sample_rate")

        if isinstance(ds, pd.Series):
            ds = int(np.unique(ds)[0])
        else:
            ds = int(ds)

        if isinstance(fs, pd.Series):
            fs = float(np.unique(fs)[0])
        else:
            fs = float(fs)

        dataset_ids.append(ds)
        sample_rates.append(fs)

    system_ids_tensor = torch.as_tensor(dataset_ids, dtype=torch.long, device=device)
    sample_f_tensor = torch.as_tensor(sample_rates, dtype=torch.float32, device=device)

    if system_ids_tensor.shape[0] == 1 and len(rows) > 1:
        system_ids_tensor = system_ids_tensor.expand(len(rows))
        sample_f_tensor = sample_f_tensor.expand(len(rows))

    return system_ids_tensor, sample_f_tensor
