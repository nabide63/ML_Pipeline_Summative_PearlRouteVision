"""Load configs/*.yaml and .env into typed settings (pydantic)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/pearlroute/config.py -> repo root is two levels above the package.
PROJECT_ROOT: Path = Path(os.environ.get("PEARLROUTE_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR: Path = PROJECT_ROOT / "configs"


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path, treating relative paths as relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class ClassConfig(BaseModel):
    """One experience class: its labelling rule, destinations and where to find images."""

    rule: str
    destinations: list[str]
    search_terms: list[str]
    fallback_terms: list[str] = []
    wikimedia_categories: list[str] = []


class PathsConfig(BaseModel):
    """File-system locations used by the data pipeline (absolute after loading)."""

    raw_dir: Path
    processed_dir: Path
    metadata_csv: Path
    manual_exclude: Path
    train_exclude: Path
    cleaned_csv: Path
    dropped_csv: Path
    splits_csv: Path
    counts_json: Path
    stats_json: Path
    demo_dir: Path
    retrain_dir: Path
    retrain_zip: Path

    @field_validator("*", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return resolve_path(value)


class SourcesConfig(BaseModel):
    """Endpoints and paging settings for the image sources."""

    user_agent: str
    openverse_api: str
    openverse_page_size: int
    wikimedia_api: str
    wikimedia_page_size: int
    wikimedia_thumb_width: int


class CollectionConfig(BaseModel):
    """How many images to fetch and how politely to fetch them."""

    target_per_class: int
    fallback_threshold: int
    per_query_limit: int
    per_category_limit: int
    request_sleep_seconds: float
    download_sleep_seconds: float
    timeout_seconds: float
    max_retries: int
    backoff_seconds: float
    max_retry_after_seconds: float
    max_side_on_save: int
    jpeg_quality: int
    openverse_licenses: list[str]
    wikimedia_subcategory_depth: int
    exclude_creators: list[str] = []      # substring match, case-insensitive
    exclude_title_words: list[str] = []   # whole-word match, case-insensitive
    min_colourfulness: float = 0.0        # below this an image counts as black-and-white; 0 disables


class CleaningConfig(BaseModel):
    """Thresholds for dropping small, corrupt and duplicate images."""

    min_short_side: int
    phash_max_distance: int
    allowed_extensions: list[str]


class HoldoutConfig(BaseModel):
    """Images kept out of every split for the demo and for retraining."""

    demo_total: int
    retrain_per_class: int


class SplitConfig(BaseModel):
    """Train/val/test ratios and grouping for a leakage-safe split."""

    train: float
    val: float
    test: float
    group_by: str
    max_class_imbalance: float


class UploadsConfig(BaseModel):
    """How photos uploaded for retraining are checked and stored (in the database)."""

    store_max_side: int = 512      # stored size; training uses 160 px, so this leaves headroom
    jpeg_quality: int = 90
    max_files: int = 300           # most photos per upload request (loose files plus zip contents)
    max_zip_mb: float = 100.0      # largest zip accepted
    max_unzipped_mb: float = 500.0 # refuse zips that expand beyond this (zip-bomb guard)


class LabelRule(BaseModel):
    """One labelling rule: photos whose text matches go to `label` (None drops them).

    Matches when the text has one of `words`, also one of `with_words` (if any), and none of `unless`.
    """

    label: str | None
    words: list[str]
    with_words: list[str] = []
    unless: list[str] = []


class DataConfig(BaseModel):
    """Everything in configs/data.yaml."""

    seed: int
    paths: PathsConfig
    sources: SourcesConfig
    classes: dict[str, ClassConfig]
    collection: CollectionConfig
    cleaning: CleaningConfig
    holdout: HoldoutConfig
    split: SplitConfig
    uploads: UploadsConfig = UploadsConfig()
    label_rules: list[LabelRule] = []     # checked in order; the first match decides the label

    @field_validator("label_rules", mode="after")
    @classmethod
    def _known_labels(cls, rules: list[LabelRule], info) -> list[LabelRule]:
        classes = set(info.data.get("classes") or {})
        unknown = {r.label for r in rules if r.label is not None} - classes
        if unknown:
            raise ValueError(f"label_rules name unknown classes: {sorted(unknown)}")
        return rules

    @property
    def class_names(self) -> list[str]:
        """Class names sorted alphabetically (the order Keras uses for directory datasets)."""
        return sorted(self.classes)


class BaselineConfig(BaseModel):
    """Model A: one Conv+MaxPool block per filters entry, then a dense layer."""

    filters: list[int] = [32, 64, 128, 128]
    dense_units: int = 128
    dropout: float = 0.5


class PearlNetConfig(BaseModel):
    """Model B: stem, then stages of residual+SE blocks (one stage per filters entry), then GAP head."""

    stem_filters: int = 32
    filters: list[int] = [64, 128, 192, 256]
    blocks: list[int] = [1, 1, 2, 1]   # residual blocks per stage; the first of each halves the size
    se_ratio: int = 8
    dropout: float = 0.3
    weight_decay: float = 1e-4


class ServingConfig(BaseModel):
    """Which saved model answers predictions, and how answers are presented."""

    active_model: str = "pearlnet"   # folder under models_dir; models/active.json (if present) overrides it
    top_k: int = 3                   # how many classes to return, most likely first
    min_confidence: float = 0.4      # below this the prediction is flagged as uncertain
    max_upload_mb: float = 10.0      # largest image the API accepts
    max_batch: int = 20              # most images per /predict/batch request


class ModelConfig(BaseModel):
    """configs/model.yaml: model input and architecture settings."""

    image_size: int  # square input side in pixels
    baseline: BaselineConfig = BaselineConfig()
    pearlnet: PearlNetConfig = PearlNetConfig()
    serving: ServingConfig = ServingConfig()


class AugmentationConfig(BaseModel):
    """Strength of each random change applied to training images; 0 / false turns one off."""

    flip_horizontal: bool = True
    rotation: float = 0.0      # fraction of a full turn, e.g. 0.04 = up to about 14 degrees either way
    zoom: float = 0.0          # fraction, e.g. 0.15 = zoom in or out by up to 15%
    translation: float = 0.0   # fraction of width/height to shift
    brightness: float = 0.0    # fraction of the 0-255 range
    contrast: float = 0.0      # fraction


class RetrainConfig(BaseModel):
    """Fine-tuning the active model on newly uploaded photos (see pearlroute.retraining)."""

    min_new_images: int = 10           # don't retrain on fewer new photos than this
    epochs: int = 8
    learning_rate: float = 1e-4        # 10x below first training: adapt without forgetting
    early_stopping_patience: int = 3
    promote_tolerance: float = 0.01    # promote if new val macro F1 >= parent's - this
    work_dir: Path = Path("data/retrain_runs")

    @field_validator("work_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return resolve_path(value)


class TrainConfig(BaseModel):
    """configs/train.yaml: batching and training-loop settings."""

    batch_size: int
    cache: bool = True  # keep decoded, resized images between epochs
    cache_dir: Path | None = None  # cache on disk here instead of in memory
    augmentation: AugmentationConfig = AugmentationConfig()
    epochs: int = 40
    learning_rate: float = 1e-3
    label_smoothing: float = 0.0   # 0.1 = target 90% on the labelled class, 10% spread over all
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-5
    models_dir: Path = Path("models")
    figures_dir: Path = Path("reports/figures")
    retrain: RetrainConfig = RetrainConfig()

    @field_validator("models_dir", "figures_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return resolve_path(value)

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _resolve_cache_dir(cls, value):
        return None if value is None else resolve_path(value)


class Settings(BaseSettings):
    """Secrets and environment-specific values, read from the environment or .env."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str | None = None
    openverse_client_id: str | None = None
    openverse_client_secret: str | None = None


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Load and validate configs/data.yaml (or the given file)."""
    return DataConfig.model_validate(_read_yaml(resolve_path(path or CONFIG_DIR / "data.yaml")))


@lru_cache
def load_model_config(path: str | Path | None = None) -> ModelConfig:
    """Load and validate configs/model.yaml (or the given file)."""
    return ModelConfig.model_validate(_read_yaml(resolve_path(path or CONFIG_DIR / "model.yaml")))


@lru_cache
def load_train_config(path: str | Path | None = None) -> TrainConfig:
    """Load and validate configs/train.yaml (or the given file)."""
    return TrainConfig.model_validate(_read_yaml(resolve_path(path or CONFIG_DIR / "train.yaml")))


@lru_cache
def get_settings() -> Settings:
    """Return the environment settings, loaded once."""
    return Settings()
