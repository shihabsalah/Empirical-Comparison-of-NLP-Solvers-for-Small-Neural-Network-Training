"""
This is the main training loop for the model.
------------------------

• epoch-style optimisers  (Adam, L-BFGS, Gradient-Descent, Trust-Region)
• one-shot optimisers     (trust-constr / other interior-point)

"""

from __future__ import annotations
from typing import Callable, List, Tuple
import time
import pathlib
import pickle

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
from optimizers.trust_region_newton_wrapper import TrustRegionNewtonWrapper


class Trainer:
    """High-level façade: prepares data, builds model, runs optimisation."""

    # --------------------------------------------------------------------- #
    #                         INITIALISATION                                #
    # --------------------------------------------------------------------- #

    def __init__(self, configuration: ExperimentConfig) -> None:
        self.configuration = configuration

        # Decide where all tensors will live (GPU or CPU)
        self.computation_device = Utils.choose_device(
            configuration.hyper_parameters.use_cuda
        )
        print(">>> Training on", self.computation_device,
              "with", configuration.optimizer_choice.name)

        # Reproducibility
        Utils.set_global_seed(configuration.hyper_parameters.seed)

        # ----------------------------- data --------------------------------
        complete_feature_array, complete_label_array = load_dataset(
            configuration.dataset, limit=None
        )
        (training_features, testing_features,
         training_labels, testing_labels) = Utils.train_test_split(
            complete_feature_array,
            complete_label_array,
            test_fraction=0.20,
            seed=configuration.hyper_parameters.seed,
        )

        (self.training_data_loader,
         self.testing_data_loader) = build_data_loaders(
            training_features,
            training_labels,
            testing_features,
            testing_labels,
            device=self.computation_device,
            batch_size=configuration.hyper_parameters.batch_size,
        )

        # ----------------------------- model -------------------------------
        self.model = build_model(configuration.model).to(
            self.computation_device)
        self.loss_function = get_loss(
            configuration.loss).to(self.computation_device)

        # -------------------------- optimiser ------------------------------
        self.optimizer: BaseOptimizer = get_optimizer(
            choice=configuration.optimizer_choice,
            model_parameters=self.model.parameters(),
            learning_rate=configuration.hyper_parameters.learning_rate,
        )

        # Misc run parameters
        self.maximum_epochs: int = configuration.hyper_parameters.epochs
        self.snapshot_interval: int = configuration.hyper_parameters.snapshot_every
        self.parameter_snapshots: List[np.ndarray] = []

    # --------------------------------------------------------------------- #
    #                       TRAINING LOOP LOGIC                             #
    # --------------------------------------------------------------------- #

    def run(self) -> None:
        """
        Entry point called from `main.py`.

        • If the optimiser is marked `is_one_shot_solver`, we call its
          `run_full_batch_solve` once.
        • Otherwise we execute a standard epoch loop.
        """
        if self.optimizer.is_one_shot_solver:
            self._run_single_call_solver()
        else:
            self._run_epoch_style_solver()

        # Persist snapshots for later PCA / visualisation
        self._save_snapshots()

    # ------------------------------------------------------------------ #
    #  A. ONE-SHOT  (trust-constr, future CMA-ES, …)                     #
    # ------------------------------------------------------------------ #

    def _run_single_call_solver(self) -> None:
        """Call optimiser once on the entire problem; then evaluate."""
        full_batch_loss_tensor = self._build_full_batch_loss_closure()
        self.optimizer.run_full_batch_solve(full_batch_loss_tensor)
        self.parameter_snapshots.append(
            Utils.flatten_params(self.model.parameters()))

        testing_loss, testing_accuracy = self._evaluate_on_testing_set()
        print(f"Final test loss  {testing_loss:.4f}   "
              f"accuracy {testing_accuracy*100:5.1f}%")

    # ------------------------------------------------------------------ #
    #  B. EPOCH-STYLE (Adam, L-BFGS, GD, Trust-Region)                   #
    # ------------------------------------------------------------------ #

    def _run_epoch_style_solver(self) -> None:
        for epoch_index in range(1, self.maximum_epochs + 1):
            training_loss, training_accuracy = self._run_single_epoch()
            testing_loss, testing_accuracy = self._evaluate_on_testing_set()

            print(f"Epoch {epoch_index:02d} | "
                  f"train loss {training_loss:.4f}  acc {training_accuracy*100:5.1f}% || "
                  f"test loss  {testing_loss:.4f}  acc {testing_accuracy*100:5.1f}%")

            if epoch_index % self.snapshot_interval == 0:
                self.parameter_snapshots.append(Utils.flatten_params(
                    self.model.parameters()
                ))

    # --------------------------------------------------------------------- #
    #                       INNER-LOOP HELPERS                               #
    # --------------------------------------------------------------------- #

    def _run_single_epoch(self) -> Tuple[float, float]:
        """Train once over the entire training set; return loss & accuracy."""
        self.model.train()
        cumulative_loss, correctly_predicted, total_examples = 0.0, 0, 0

        for input_batch_of_images, target_digit_labels in self.training_data_loader:

            # 1) Build the closure (needed by L-BFGS, harmless for others)
            def closure() -> torch.Tensor:
                self.optimizer.zero_grad()
                predicted_logits = self.model(input_batch_of_images)
                batch_loss_tensor = self.loss_function(
                    predicted_logits, target_digit_labels
                )
                # compute gradients only for optimizers expecting internal backward
                if not isinstance(self.optimizer, TrustRegionNewtonWrapper):
                    batch_loss_tensor.backward()
                return batch_loss_tensor

            # 2) One optimiser step
            self.optimizer.step(closure)

            # 3) Metrics for this mini-batch (no grad)
            with torch.no_grad():
                predicted_logits = self.model(input_batch_of_images)
                cumulative_loss += self.loss_function(
                    predicted_logits, target_digit_labels
                ).item() * input_batch_of_images.size(0)

                if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
                    batch_predictions = predicted_logits.argmax(dim=1)
                    correctly_predicted += (
                        (batch_predictions == target_digit_labels).sum().item()
                    )
                total_examples += input_batch_of_images.size(0)

        average_loss = cumulative_loss / total_examples
        accuracy = correctly_predicted / total_examples
        return average_loss, accuracy

    # ------------------------------------------------------------------ #

    def _evaluate_on_testing_set(self) -> Tuple[float, float]:
        """Return (loss, accuracy) on held-out test set."""
        self.model.eval()
        cumulative_loss, correctly_predicted, total_examples = 0.0, 0, 0
        with torch.no_grad():
            for input_batch_of_images, target_digit_labels in self.testing_data_loader:
                predicted_logits = self.model(input_batch_of_images)
                cumulative_loss += self.loss_function(
                    predicted_logits, target_digit_labels
                ).item() * input_batch_of_images.size(0)

                if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
                    batch_predictions = predicted_logits.argmax(dim=1)
                    correctly_predicted += (
                        (batch_predictions == target_digit_labels).sum().item()
                    )
                total_examples += input_batch_of_images.size(0)

        return cumulative_loss / total_examples, correctly_predicted / total_examples

    # ------------------------------------------------------------------ #

    def _build_full_batch_loss_closure(self) -> Callable[[], torch.Tensor]:
        """
        Returns a callable that recomputes the full-dataset loss **as a tensor**.
        One-shot solvers will call this many times.
        """
        def closure() -> torch.Tensor:
            self.model.train()
            running_total, mini_batches = 0.0, 0
            for feature_tensor, label_tensor in self.training_data_loader:
                running_total += self.loss_function(
                    self.model(feature_tensor), label_tensor
                )
                mini_batches += 1
            return running_total / mini_batches
        return closure

    # ------------------------------------------------------------------ #
    #                       SNAPSHOT PERSISTENCE                          #
    # ------------------------------------------------------------------ #

    def _save_snapshots(self) -> None:
        optimizer_name = self.configuration.optimizer_choice.name
        target_directory = pathlib.Path("runs") / optimizer_name
        target_directory.mkdir(parents=True, exist_ok=True)
        with open(target_directory / "snapshots.pkl", "wb") as handle:
            pickle.dump(self.parameter_snapshots, handle)
        print(f"Snapshots saved to {target_directory / 'snapshots.pkl'}")
