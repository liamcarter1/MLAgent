"""Data stage: obtain raw tabular or image data, then hand the user profile.py to run."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mlagent.datasources.drive import list_candidates
from mlagent.datasources.hf import search_datasets
from mlagent.imageset import DEFAULT_IMAGE_SIZE, IMAGE_SIZES
from mlagent.modality import modality_for
from mlagent.profile import profile_markdown
from mlagent.stages.base import Handoff, ScriptStageBase, StageContext
from mlagent.synth.images import MAX_CLASSES as MAX_IMAGE_CLASSES
from mlagent.synth.images import SynthImageConfig
from mlagent.synth.images import generate as generate_images
from mlagent.synth.tabular import TARGET, SynthTabularConfig, generate
from mlagent.templates_io import copy_shared

RAW_FILE = "data.csv"
IMAGE_RAW_FILE = "data.npz"
IMAGE_TARGET = "label"
IMAGE_N_CHANNELS = 3
META_FILE = "data_meta.json"
PROFILE_RAW_FILE = "profile_raw.json"
DEFAULT_QUIRKS = ("missing", "duplicates", "id_column", "categorical", "whitespace", "outliers")
DEFAULT_SEARCH_ROOTS = (Path("/content/drive/MyDrive"), Path("/content"))
TABULAR_TASKS = {"tabular_classification": "classification", "tabular_regression": "regression"}
# Preference order: an exact or suffix match on an earlier keyword beats a later one.
TARGET_GUESS_KEYWORDS = (
    "target", "label", "class", "y", "outcome", "result", "churn", "price", "score",
)


def guess_target(df: pd.DataFrame) -> str | None:
    """The column most likely to be the target: the first column (by keyword preference
    order) whose lowercased name is one of `TARGET_GUESS_KEYWORDS` or ends with one, else
    the last column. `None` only for a frame with no columns."""
    columns = [str(c) for c in df.columns]
    if not columns:
        return None
    for keyword in TARGET_GUESS_KEYWORDS:
        for col in columns:
            lower = col.lower()
            if lower == keyword or lower.endswith(keyword):
                return col
    return columns[-1]


class DataStage(ScriptStageBase):
    name = "data"

    def __init__(
        self, search_roots=None, hf_search=search_datasets, hf_load=None, hf_image_load=None
    ):
        self.search_roots = (
            list(search_roots) if search_roots is not None else list(DEFAULT_SEARCH_ROOTS)
        )
        self.hf_search = hf_search
        # None means "whatever the task type's modality record says"; tests inject a fake.
        self.hf_load = hf_load
        self.hf_image_load = hf_image_load

    def is_complete(self, ctx: StageContext) -> bool:
        meta = ctx.project.read_json(META_FILE)
        if not meta or not meta.get("target"):
            return False
        data_file = IMAGE_RAW_FILE if meta.get("modality") == "image" else RAW_FILE
        return (
            (ctx.project.data_raw / data_file).exists()
            and ctx.project.exists(PROFILE_RAW_FILE)
        )

    def prepare(self, ctx: StageContext) -> Handoff:
        spec = ctx.spec()
        modality = modality_for(spec.task_type)
        ctx.teaching().preamble("data", {"spec": spec.to_dict()})
        if modality.name == "image":
            meta = self._image(ctx, spec, modality)
        else:
            meta = self._tabular(ctx, spec, modality)
        ctx.project.write_json(META_FILE, meta)
        copy_shared(modality.profile_template, ctx.project.root)
        return Handoff(stage=self.name, commands=[["profile.py"]], outputs=[PROFILE_RAW_FILE])

    def _tabular(self, ctx: StageContext, spec, modality) -> dict:
        if spec.data_source == "synthetic":
            df, target, meta = self._synthetic(ctx, TABULAR_TASKS[spec.task_type])
        elif spec.data_source == "drive":
            df, target, meta = self._drive(ctx, modality)
        else:
            df, target, meta = self._huggingface(ctx, modality)

        ctx.project.data_raw.mkdir(parents=True, exist_ok=True)
        path = ctx.project.data_raw / RAW_FILE
        df.to_csv(path, index=False)
        meta.update(
            {
                "target": target,
                "raw_path": path.relative_to(ctx.project.root).as_posix(),
                "raw_n_rows": int(len(df)),
                "raw_n_cols": int(df.shape[1]),
                "task_type": spec.task_type,
            }
        )
        ctx.display(
            f"I saved {len(df)} rows and {df.shape[1]} columns to `data/raw/data.csv` and wrote "
            "`profile.py`, which measures the data and draws four figures. Run it in the next "
            "cell; nothing about the raw data is changed."
        )
        return meta

    def debrief(self, ctx: StageContext) -> None:
        profile = ctx.project.read_json(PROFILE_RAW_FILE)
        if not isinstance(profile, dict):
            ctx.display(
                "I can't see `profile_raw.json` yet. Run the `profile.py` cell, then run this "
                "cell again."
            )
            return
        ctx.display(profile_markdown(profile))
        figures = [
            ctx.project.plots_dir / str(name) for name in (profile.get("figures") or [])
        ]
        payload = {
            "spec": ctx.spec().to_dict(),
            "profile": {k: v for k, v in profile.items() if k != "figures"},
        }
        fallback = "Data saved. Next: the [[data cleaning]] audit."
        ctx.display(ctx.teaching().debrief("data", payload, figures, fallback=fallback))

    def _synthetic(self, ctx: StageContext, task: str):
        q = ctx.questioner
        n_samples = int(q.number("How many rows?", default=1000, minimum=100, maximum=200000,
                                 key="data.n_rows"))
        n_features = int(q.number("How many numeric features?", default=8, minimum=2,
                                  maximum=100, key="data.n_features"))
        n_classes, class_balance = 2, 0.5
        if task == "classification":
            n_classes = int(q.number("How many classes?", default=2, minimum=2, maximum=10,
                                     key="data.n_classes"))
            if n_classes == 2:
                class_balance = q.number(
                    "Fraction of rows in the majority class (0.5 = balanced)?",
                    default=0.5, minimum=0.5, maximum=0.95, key="data.class_balance",
                )
        noise = q.number(
            "Label/measurement noise (0 = clean, 0.3 = very noisy)?",
            default=0.1, minimum=0.0, maximum=1.0, key="data.noise",
        )
        inject = q.confirm(
            "Inject realistic data problems (missing values, duplicates, an ID column, messy "
            "categories, outliers) so the cleaning stage has work to do?",
            default=True, key="data.inject_quirks",
        )
        cfg = SynthTabularConfig(
            task=task, n_samples=n_samples, n_features=n_features, n_classes=n_classes,
            class_balance=class_balance, noise=noise, seed=42,
            quirks=DEFAULT_QUIRKS if inject else (),
        )
        df = generate(cfg)
        return df, TARGET, {"source": "synthetic", "synth_config": asdict(cfg)}

    def _ask_target(self, ctx: StageContext, df: pd.DataFrame) -> str:
        cols = [str(c) for c in df.columns]
        guess = guess_target(df)
        if guess is not None:
            ctx.display(
                f"My guess is `{guess}` (it looks like a label column); pick a different "
                "one if I am wrong."
            )
        return ctx.questioner.choice(
            "Which column is the target (what you want to predict)?", cols, allow_other=False,
            key="data.target_column", default=guess,
        )

    def _drive(self, ctx: StageContext, modality):
        q = ctx.questioner
        candidates = list_candidates(self.search_roots)
        if candidates:
            answer = q.choice(
                "Which file holds your data? (pick one or type a full path)",
                [str(p) for p in candidates], allow_other=True, key="data.drive_path",
            )
        else:
            answer = q.text(
                "No CSV/Parquet/Excel files found. Enter the full path to your data file",
                key="data.drive_path",
            )
        path = Path(answer.strip())
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        df = modality.load_drive(path)
        target = self._ask_target(ctx, df)
        return df, target, {"source": "drive", "source_path": str(path)}

    def _huggingface(self, ctx: StageContext, modality):
        q = ctx.questioner
        results = []
        for _ in range(3):
            query = q.text(
                "Describe the dataset you want (a few keywords, e.g. 'credit card fraud')",
                key="data.hf_query",
            )
            results = self.hf_search(query)
            if results:
                break
            ctx.display(f"No datasets found for '{query}'. Try different words.")
        if not results:
            raise RuntimeError("no HuggingFace datasets found after 3 searches")
        labels = [f"{r.id} — {r.downloads:,} downloads — {r.description[:60]}" for r in results]
        pick = q.choice("Which dataset?", labels, allow_other=False)
        chosen = results[labels.index(pick)]
        ctx.display(f"Downloading **{chosen.id}** from the HuggingFace Hub...")
        loader = self.hf_load or modality.load_hf
        df = loader(chosen.id)
        target = self._ask_target(ctx, df)
        return df, target, {"source": "huggingface", "hf_id": chosen.id}

    def _image(self, ctx: StageContext, spec, modality) -> dict:
        q = ctx.questioner
        image_size = int(q.choice(
            "How big should each image be, in pixels? (bigger sees more detail but trains "
            "more slowly)",
            [str(s) for s in IMAGE_SIZES], allow_other=False, key="data.image_size",
            default=str(DEFAULT_IMAGE_SIZE),
        ))
        if spec.data_source == "synthetic":
            imageset, source_meta = self._image_synthetic(ctx, image_size)
            skipped: list[tuple[str, str]] = []
        elif spec.data_source == "drive":
            imageset, skipped, source_meta = self._image_drive(ctx, modality, image_size)
        else:
            imageset, source_meta = self._image_huggingface(ctx, modality, image_size)
            skipped = []

        modality.write_raw(imageset, ctx.project.data_raw)
        meta = {
            **source_meta,
            "target": IMAGE_TARGET,
            "task_type": spec.task_type,
            "modality": "image",
            "raw_path": f"data/raw/{IMAGE_RAW_FILE}",
            "raw_n_rows": imageset.n_images,
            "raw_n_cols": imageset.image_size * imageset.image_size * IMAGE_N_CHANNELS,
            "image_size": imageset.image_size,
            "n_channels": IMAGE_N_CHANNELS,
            "n_classes": len(imageset.class_names),
            "class_labels": list(imageset.class_names),
            "skipped_files": [[str(p), str(r)] for p, r in skipped],
        }
        note = (
            f" I skipped {len(skipped)} file(s) that were not readable images; the audit "
            "lists them." if skipped else ""
        )
        message = (
            f"I saved {imageset.n_images} images at {imageset.image_size}x"
            f"{imageset.image_size} pixels across {len(imageset.class_names)} classes "
            f"({', '.join(imageset.class_names)}) to `data/raw/{IMAGE_RAW_FILE}`, with one "
            f"row per image in `data/raw/manifest.csv`.{note} I also wrote `profile.py`, "
            "which measures the images and draws four figures. Run it in the next cell; "
            "nothing about the raw images is changed."
        )
        ctx.display(message)
        return meta

    def _image_synthetic(self, ctx: StageContext, image_size: int):
        q = ctx.questioner
        n_images = int(q.number("How many images?", default=300, minimum=20, maximum=20000,
                                key="data.n_images"))
        n_classes = int(q.number(
            f"How many shape classes? (2 to {MAX_IMAGE_CLASSES})", default=3, minimum=2,
            maximum=MAX_IMAGE_CLASSES, key="data.n_classes",
        ))
        noise = q.number(
            "Background noise (0 = clean shapes, 0.3 = grainy and realistic)?",
            default=0.1, minimum=0.0, maximum=1.0, key="data.noise",
        )
        inject = q.confirm(
            "Inject realistic problems (duplicate images, blank images, an uneven number "
            "per class) so the cleaning stage has work to do?",
            default=True, key="data.inject_quirks",
        )
        cfg = SynthImageConfig(
            n_images=n_images, image_size=image_size, n_classes=n_classes, seed=42,
            noise=float(noise),
            class_imbalance=0.55 if inject else 0.0,
            duplicate_fraction=0.08 if inject else 0.0,
            blank_fraction=0.05 if inject else 0.0,
        )
        return generate_images(cfg), {"source": "synthetic", "synth_config": asdict(cfg)}

    def _image_drive(self, ctx: StageContext, modality, image_size: int):
        q = ctx.questioner
        answer = q.text(
            "Which folder holds your images? It needs one subfolder per class, e.g. "
            "`MyDrive/pets/cats/` and `MyDrive/pets/dogs/`",
            key="data.drive_folder",
        )
        folder = Path(answer.strip())
        if not folder.is_dir():
            raise NotADirectoryError(f"no such folder: {folder}")
        cap = int(q.number(
            "Cap the number of images? (0 means use them all; a cap keeps the first run "
            "quick)", default=0, minimum=0, maximum=100000, key="data.max_images",
        ))
        ctx.display(f"Reading images from **{folder}**; this is the slow part, once.")
        imageset, skipped = modality.load_drive(folder, image_size, cap or None)
        return imageset, skipped, {"source": "drive", "source_path": str(folder)}

    def _image_huggingface(self, ctx: StageContext, modality, image_size: int):
        q = ctx.questioner
        dataset_id = q.text(
            "Which HuggingFace image dataset? Give its id, e.g. `cifar10` or "
            "`beans`", key="data.hf_dataset",
        ).strip()
        cap = int(q.number(
            "Cap the number of images? (0 means use them all)", default=2000, minimum=0,
            maximum=100000, key="data.max_images",
        ))
        ctx.display(f"Downloading **{dataset_id}** from the HuggingFace Hub...")
        loader = self.hf_image_load or modality.load_hf
        imageset = loader(dataset_id, image_size, max_images=cap or None)
        return imageset, {"source": "huggingface", "hf_id": dataset_id}
