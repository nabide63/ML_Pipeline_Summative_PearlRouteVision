"""End-to-end: load data, build, train, evaluate and save one model.

Writes to <models_dir>/<name>/:
    model.keras, metadata.json   the trained model and what prediction needs (see models.io)
    history.csv                  loss/accuracy per epoch
    metrics.json                 validation and test metrics
    checkpoints/best.keras       best model so far (git-ignored; useful if a run is interrupted)
and <figures_dir>/<name>_learning_curves.png, <name>_confusion_matrix.png.
"""

from __future__ import annotations

import json
import time
from typing import Any

import keras
import matplotlib.pyplot as plt

from pearlroute.config import DataConfig, ModelConfig, TrainConfig
from pearlroute.data.loaders import load_class_weights, load_datasets
from pearlroute.evaluation.metrics import evaluate_model
from pearlroute.evaluation.plots import plot_confusion_matrix, plot_history
from pearlroute.logging_utils import get_logger
from pearlroute.models.io import save_model
from pearlroute.models.registry import build_model
from pearlroute.preprocessing.augmentation import make_augment
from pearlroute.preprocessing.stats import load_stats
from pearlroute.training.trainer import train_model

log = get_logger(__name__)


def train_and_evaluate(name: str, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig,
                       epochs: int | None = None, augment: bool = True, run_name: str | None = None
                       ) -> dict[str, Any]:
    """Train model `name` on the train split, pick the best epoch on val, score it on val and test.

    Results go to models/<run_name or name>/, so variants (e.g. "pearlnet_v2") can sit side by side.
    Returns {"model", "history", "val", "test", "out_dir", "seconds"}.
    """
    run_name = run_name or name
    keras.utils.set_random_seed(data_cfg.seed)
    out_dir = train_cfg.models_dir / run_name
    stats = load_stats(data_cfg.paths.stats_json)
    augment_fn = make_augment(train_cfg.augmentation, seed=data_cfg.seed) if augment else None
    sets = load_datasets(data_cfg, model_cfg, train_cfg, augment=augment_fn, stats=stats)

    model = build_model(name, model_cfg, len(data_cfg.class_names))
    log.info("%s (%s): %s parameters, training up to %d epochs", run_name, name, f"{model.count_params():,}",
             epochs or train_cfg.epochs)
    start = time.time()
    history = train_model(model, sets["train"], sets["val"], train_cfg, out_dir,
                          class_weight=load_class_weights(data_cfg), epochs=epochs)
    seconds = round(time.time() - start, 1)

    val = evaluate_model(model, sets["val"], data_cfg.class_names)
    test = evaluate_model(model, sets["test"], data_cfg.class_names)
    best_epoch = int(history["val_loss"].idxmin())
    summary = {"best_epoch": best_epoch, "epochs_run": int(len(history)), "train_seconds": seconds,
               "parameters": int(model.count_params()), "augmentation": augment, "architecture": name,
               "settings": {"model": getattr(model_cfg, name).model_dump(),
                            "train": train_cfg.model_dump(include={"learning_rate", "label_smoothing", "batch_size"}),
                            "augmentation": train_cfg.augmentation.model_dump() if augment else None}}
    (out_dir / "metrics.json").write_text(json.dumps({**summary, "val": val, "test": test}, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    mean, std = stats
    save_model(model, out_dir, data_cfg.class_names, model_cfg.image_size, list(mean), list(std),
               test_accuracy=test["accuracy"], test_f1_macro=test["f1_macro"], **summary)

    figures = train_cfg.figures_dir
    figures.mkdir(parents=True, exist_ok=True)
    for fig, file in [(plot_history(history, f"{run_name}: learning curves"), f"{run_name}_learning_curves.png"),
                      (plot_confusion_matrix(test["confusion_matrix"], data_cfg.class_names,
                                             f"{run_name}: test set confusion matrix"), f"{run_name}_confusion_matrix.png")]:
        fig.savefig(figures / file, dpi=120)
        plt.close(fig)

    log.info("%s: best epoch %d of %d, %.0fs | val acc %.3f | test acc %.3f, macro F1 %.3f, ROC-AUC %.3f",
             run_name, best_epoch, len(history), seconds, val["accuracy"], test["accuracy"], test["f1_macro"],
             test["roc_auc_macro"])
    return {"model": model, "history": history, "val": val, "test": test, "out_dir": out_dir, "seconds": seconds}
