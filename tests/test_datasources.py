from types import SimpleNamespace

import pandas as pd
import pytest

from mlagent.datasources import drive, hf


def test_list_candidates_respects_depth_skips_and_limit(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "deep" / "er" / "est").mkdir(parents=True)
    (tmp_path / "deep" / "er" / "est" / "far.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "deep" / "near.parquet").write_bytes(b"")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hidden.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("no", encoding="utf-8")
    found = drive.list_candidates([tmp_path, tmp_path / "missing"], max_depth=2)
    assert [p.name for p in found] == ["a.csv", "near.parquet"]
    assert drive.list_candidates([tmp_path], limit=1) == [tmp_path / "a.csv"]


def test_load_table_by_extension(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    tsv = tmp_path / "t.tsv"
    tsv.write_text("a\tb\n1\t2\n", encoding="utf-8")
    pq = tmp_path / "t.parquet"
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(pq)
    for path in (csv, tsv, pq):
        assert drive.load_table(path).to_dict("records") == [{"a": 1, "b": 2}]
    with pytest.raises(ValueError):
        drive.load_table(tmp_path / "t.xyz")


class FakeApi:
    def __init__(self):
        self.calls = []

    def list_datasets(self, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                id="org/churn", downloads=1200, likes=5,
                tags=["task:tabular", "csv"], description=None
            ),
            SimpleNamespace(
                id="org/other", downloads=None, likes=None, tags=None,
                description="Some data"
            ),
        ]


def test_search_datasets_maps_results():
    api = FakeApi()
    results = hf.search_datasets("churn", limit=2, api=api)
    assert api.calls[0]["search"] == "churn" and api.calls[0]["limit"] == 2
    assert results[0] == hf.HFDataset(id="org/churn", downloads=1200, likes=5, description="csv")
    assert results[1].downloads == 0 and results[1].description == "Some data"


def test_load_tabular_with_split_and_dict_fallback():
    frame = pd.DataFrame({"a": [1]})

    def loader_ok(dataset_id, **kwargs):
        assert kwargs == {"split": "train"}
        return SimpleNamespace(to_pandas=lambda: frame)

    assert hf.load_tabular("x/y", loader=loader_ok).equals(frame)

    class DictLike(dict):
        pass

    def loader_dict(dataset_id, **kwargs):
        if "split" in kwargs:
            raise ValueError("bad split")
        return DictLike(validation=SimpleNamespace(to_pandas=lambda: frame))

    assert hf.load_tabular("x/y", loader=loader_dict).equals(frame)
