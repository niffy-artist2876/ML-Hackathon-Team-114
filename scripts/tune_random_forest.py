"""Tune the Random Forest with Optuna TPE on the fixed trip folds."""

from __future__ import annotations

import argparse
from hashlib import file_digest
from itertools import product
import json
from pathlib import Path
from random import Random
import sys
import time

import numpy as np
import optuna
import pandas as pd
import sklearn
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_candidate_model import load_candidate_tables, load_folds, resolve_path, write_json
from src.model.random_forest import build_random_forest
from src.model.tuning import prepare_folds


DONE = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def evaluated_trials(study):
    return [
        trial for trial in study.get_trials(deepcopy=False, states=DONE)
        if "duplicate_of" not in trial.user_attrs
    ]


def make_objective(prepared, baseline_parameters, search_space):
    def objective(trial):
        parameters = baseline_parameters | {
            name: trial.suggest_categorical(name, choices) for name, choices in search_space.items()
        }
        for existing in evaluated_trials(trial.study):
            if existing.params == trial.params:
                trial.set_user_attr("duplicate_of", existing.number)
                raise optuna.TrialPruned(f"Duplicate of trial {existing.number}")
        correct, points = 0, 0
        for step, fold in enumerate(prepared):
            started = time.perf_counter()
            forest = build_random_forest(parameters).named_steps["classifier"]
            forest.fit(fold.x_train, fold.y_train)
            probability = np.clip(forest.predict_proba(fold.x_validation)[:, 1], 1e-12, 1 - 1e-12)
            fold_correct = fold.correct_count(probability)
            correct += fold_correct
            points += fold.point_count
            seconds = time.perf_counter() - started
            trial.set_user_attr(f"fold_{fold.fold}_correct", fold_correct)
            trial.set_user_attr(f"fold_{fold.fold}_seconds", seconds)
            trial.report(correct / points, step=step)
            print(
                f"[trial {trial.number}] fold {fold.fold}: "
                f"{fold_correct / fold.point_count:.2%} Top-1 in {seconds:.1f}s",
                flush=True,
            )
            if step + 1 < len(prepared) and trial.should_prune():
                raise optuna.TrialPruned()
        return correct / points

    return objective


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "random_forest_tuning.json")
    parser.add_argument("--trials", type=int, help="Total trial budget including existing trials")
    parser.add_argument("--timeout-seconds", type=float, help="Stop starting trials after this duration")
    parser.add_argument(
        "--recover-interrupted", action="store_true",
        help="Mark orphaned RUNNING trials failed after confirming the previous process stopped",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    budget = config["trials"] if args.trials is None else args.trials
    if budget < 1 or (args.timeout_seconds is not None and args.timeout_seconds <= 0):
        parser.error("Trial budget and timeout must be positive")
    baseline = json.loads(resolve_path(config["baseline_config"]).read_text(encoding="utf-8"))
    baseline_parameters = baseline["model_parameters"]
    search_space = config["search_space"]
    for name, values in search_space.items():
        if baseline_parameters[name] not in values:
            raise ValueError(f"Baseline value for {name} must be in the search space")
    configurations = list(product(*search_space.values()))
    if budget > len(configurations):
        parser.error(f"Trial budget exceeds the {len(configurations)} unique configurations")

    out = resolve_path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fold_file = resolve_path(baseline["folds_path"])
    train_paths = sorted((REPO / "data" / "processed" / "train").glob("*/candidates.parquet"))
    protocol = {
        "baseline": baseline,
        "search": {key: value for key, value in config.items() if key != "trials"},
        "folds_sha256": file_hash(fold_file),
        "train_tables_sha256": {path.parent.name: file_hash(path) for path in train_paths},
        "versions": {"optuna": optuna.__version__, "scikit_learn": sklearn.__version__, "pandas": pd.__version__},
    }
    startup = config["startup_trials"]
    journal_path = str(out / "study.journal")
    study = optuna.create_study(
        study_name="random_forest_point_top1",
        storage=JournalStorage(JournalFileBackend(journal_path, lock_obj=JournalFileOpenLock(journal_path))),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["seed"], n_startup_trials=startup),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=startup, n_warmup_steps=config["prune_after_folds"] - 1,
        ),
        load_if_exists=True,
    )
    saved_protocol = study.user_attrs.get("protocol")
    if saved_protocol is None:
        study.set_user_attr("protocol", protocol)
    elif saved_protocol != protocol:
        raise ValueError("Study inputs or protocol changed; choose a fresh output directory")
    running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
    if running and not args.recover_interrupted:
        raise RuntimeError("A study trial is still running; confirm the previous process stopped before recovery")
    for trial in running:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

    def export(study, _trial=None):
        study.trials_dataframe().to_csv(out / "trials.csv", index=False)
        complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if complete:
            best = study.best_trial
            write_json(out / "best_trial.json", {
                "trial": best.number, "validation_top1": best.value, "parameters": best.params,
                "complete_trials": len(complete),
                "pruned_trials": len(evaluated_trials(study)) - len(complete),
                "duplicate_proposals": sum("duplicate_of" in trial.user_attrs for trial in study.trials),
            })

    if not study.trials:
        study.enqueue_trial({name: baseline_parameters[name] for name in search_space})
    if len(evaluated_trials(study)) < budget:
        tables = load_candidate_tables("train")
        folds = load_folds(fold_file, set(tables))
        if len(set(folds.values())) != 5:
            raise ValueError("Expected the five shared trip folds")
        prepared = prepare_folds(tables, folds, baseline)
        del tables
        objective = make_objective(prepared, baseline_parameters, search_space)
        started = time.monotonic()
        random = Random(config["seed"])
        duplicates = 0
        while len(evaluated_trials(study)) < budget:
            if args.timeout_seconds is not None and time.monotonic() - started >= args.timeout_seconds:
                break
            if duplicates >= 5:
                seen = {tuple(trial.params[name] for name in search_space) for trial in evaluated_trials(study)}
                unseen = [params for params in configurations if params not in seen]
                study.enqueue_trial(
                    dict(zip(search_space, random.choice(unseen), strict=True)),
                    user_attrs={"proposal": "unseen_exploration"},
                )
                duplicates = 0
            study.optimize(objective, n_trials=1, n_jobs=1, callbacks=[export])
            duplicates = duplicates + 1 if "duplicate_of" in study.trials[-1].user_attrs else 0
    export(study)
    baseline_trial = study.trials[0]
    if baseline_trial.state != optuna.trial.TrialState.COMPLETE:
        raise RuntimeError("The baseline trial must finish before exporting the tuned configuration")
    best = study.best_trial
    tuned = baseline | {
        "model_parameters": baseline_parameters | best.params,
        "outputs": {
            "directory": (out / "evaluation").as_posix(),
            "model_path": "models/random_forest_tuned/model.joblib",
        },
    }
    if out.is_relative_to(REPO):
        tuned["outputs"]["directory"] = (out / "evaluation").relative_to(REPO).as_posix()
    write_json(out / "tuned_config.json", tuned)
    print(f"Baseline {baseline_trial.value:.4%}; best trial {best.number}: {best.value:.4%} {best.params}")


if __name__ == "__main__":
    main()
