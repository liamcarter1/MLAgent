"""Search and load tabular datasets from HuggingFace Hub (lazy imports, injectable)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class HFDataset:
    id: str
    downloads: int
    likes: int
    description: str


def search_datasets(
    query: str, limit: int = 8, api=None
) -> list[HFDataset]:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    results = api.list_datasets(
        search=query, sort="downloads", direction=-1, limit=limit
    )
    out: list[HFDataset] = []
    for item in results:
        dataset_id = getattr(item, "id", None)
        if dataset_id is None:
            continue
        description = getattr(item, "description", None) or ""
        if not description:
            tags = getattr(item, "tags", None) or []
            description = ", ".join(t for t in tags if ":" not in t)
        out.append(
            HFDataset(
                id=str(dataset_id),
                downloads=int(getattr(item, "downloads", 0) or 0),
                likes=int(getattr(item, "likes", 0) or 0),
                description=description[:120],
            )
        )
    return out


def load_tabular(
    dataset_id: str, split: str = "train", config: str | None = None, loader=None
) -> pd.DataFrame:
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    kwargs = {"name": config} if config else {}
    try:
        ds = loader(dataset_id, split=split, **kwargs)
    except ValueError as first:
        try:
            ds = loader(dataset_id, **kwargs)
        except Exception as second:
            raise second from first
    if hasattr(ds, "keys") and not hasattr(ds, "to_pandas"):
        key = "train" if "train" in ds else next(iter(ds.keys()))
        ds = ds[key]
    return ds.to_pandas()
