from __future__ import annotations
import time
import pathlib
import pickle
from typing import Callable, List, Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np

from config import ExperimentConfig
from utils import Utils
from data import load_dataset
from models import build_model
from losses import get_loss
from optimizers import get_optimizer, BaseOptimizer
from trainer.data_pipeline import build_data_loaders


class TrainingLoop:
    """
    Orchestrates one experiment: prepares data, builds model & optimiser,
    runs for N epochs (or calls a one-shot solver), stores snapshots.
    """

    def __init__(self, configuration: ExperimentConfig) -> None:
        self.config = configuration
        self.device = Utils.choose_device(
            configuration.hyper_parameters.use_cuda)
        Utils.set_global_seed(configuration.hyper_parameters.seed)

        # ---- data ------------------------------------------------------------
        features, labels = load_dataset(configuration.dataset, limit=None)
        split = Utils.train_test_split(
            features, labels, test_fraction=0.2,
            seed=configuration.hyper_parameters.seed)
        (tr_X, te_X, tr_y, te_y) = split
        self.train_loader, self.test_loader = build_data_loaders(
            tr_X, tr_y, te_X, te_y,
            device=self.device,
            batch_size=configuration.hyper_parameters.batch_size)

        # ---- model, loss, optimiser -----------------------------------------
        self.model = build_model(configuration.model).to(self.device)
        self.loss_function = get_loss(configuration.loss).to(self.device)
        self.optimizer: BaseOptimizer = get_optimizer(
            choice=configuration.optimizer_choice,
            model_parameters=self.model.parameters(),
            learning_rate=configuration.hyper_parameters.learning_rate,
        )
        self.num_epochs = configuration.hyper_parameters.epochs
        self.snapshot_interval = configuration.hyper_parameters.snapshot_every

        self.snapshots: List[np.ndarray] = []

    # -------------------------------------------------------------------------
    def _single_epoch(self) -> Tuple[float, float]:
        """Train once over train_loader; return (loss, accuracy)."""
        self.model.train()
        total, correct, running_loss = 0, 0, 0.0

        for features, labels in self.train_loader:
            def closure() -> torch.Tensor:
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                return loss

            # Some optimisers need a closure, others ignore it
            self.optimizer.step(closure)      # works for Adam, LBFGS, etc.

            with torch.no_grad():
                outputs = self.model(features)
                running_loss += self.loss_function(outputs,
                                                   labels).item() * features.size(0)
                if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += features.size(0)

        return running_loss / total, correct / total

    def _evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total, correct, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for features, labels in self.test_loader:
                outputs = self.model(features)
                running_loss += self.loss_function(outputs,
                                                   labels).item() * features.size(0)
                if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += features.size(0)
        return running_loss / total, correct / total

    # -------------------------------------------------------------------------
    def run(self) -> None:
        """
        If the optimiser implements `run_full_batch_solve`, that method is
        invoked once; otherwise we fall back to epoch loop.
        """
        print(
            f">>> Training on {self.device} with {self.config.optimizer_choice.name}")

        # trainer/training_loop.py  (inside run())
        if self.optimizer.is_one_shot_solver:
            # ------------------------------------------------ one-shot path
            full_loss = self._build_full_batch_closure()
            self.optimizer.run_full_batch_solve(full_loss)
            self.snapshots.append(
                Utils.flatten_params(self.model.parameters()))
            test_loss, test_acc = self._evaluate()
            print(
                f"Final test   loss {test_loss:.4f}  acc {test_acc*100:.1f}%")

        else:
            # ------------------------------------------------ epoch loop
            for epoch in range(1, self.num_epochs + 1):
                train_loss, train_acc = self._single_epoch()
                test_loss,  test_acc = self._evaluate()
                print(f"Epoch {epoch:02d} | "
                      f"train loss {train_loss:.4f} acc {train_acc*100:5.1f}% || "
                      f"test loss  {test_loss:.4f} acc {test_acc*100:5.1f}%")
                if epoch % self.snapshot_interval == 0:
                    self.snapshots.append(
                        Utils.flatten_params(self.model.parameters()))

        self._save_snapshots()

    # -------------------------------------------------------------------------
    def _build_full_batch_closure(self) -> Callable[[], float]:
        """Returns a function that computes the full-batch loss once."""
        def closure() -> float:
            self.model.train()
            running, total = 0.0, 0
            for features, labels in self.train_loader:
                outputs = self.model(features)
                running += self.loss_function(outputs,
                                              labels).item() * features.size(0)
                total += features.size(0)
            return running / total
        return closure

    # -------------------------------------------------------------------------
    def _save_snapshots(self) -> None:
        run_dir = pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "snapshots.pkl", "wb") as fh:
            pickle.dump(self.snapshots, fh)
        print(f"Snapshots saved to {run_dir/'snapshots.pkl'}")
