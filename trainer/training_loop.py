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
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

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

        # Early stopping parameters
        self.early_stop_patience: int = 5  # Stop if no improvement for this many epochs
        # Minimum change to count as improvement
        self.early_stop_min_delta: float = 1e-4
        self.best_val_loss: float = float('inf')
        self.patience_counter: int = 0

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
        # Reset peak GPU memory stats if using CUDA
        if self.computation_device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.computation_device)
        # Time the solver
        start_time = time.perf_counter()
        if self.optimizer.is_one_shot_solver:
            self._run_single_call_solver()
        else:
            self._run_epoch_style_solver()
        end_time = time.perf_counter()
        # Compute metrics
        runtime_s = end_time - start_time
        if self.computation_device.type == 'cuda':
            peak_mem_bytes = torch.cuda.max_memory_allocated(
                self.computation_device)
        else:
            import psutil
            peak_mem_bytes = psutil.Process().memory_info().rss
        peak_mem_mb = peak_mem_bytes / (1024**2)
        # Persist snapshots and metrics
        metrics = {'runtime_s': runtime_s, 'peak_mem_mb': peak_mem_mb}
        self._save_snapshots(metrics)

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
        """Run over multiple epochs with progress bar updating in place."""
        # Create progress bar for epochs
        pbar = tqdm(range(1, self.maximum_epochs + 1),
                    desc="Training", leave=True, position=0)

        # Collect results without printing each epoch
        results = []

        for epoch_index in pbar:
            training_loss, training_accuracy = self._run_single_epoch()
            testing_loss, testing_accuracy = self._evaluate_on_testing_set()

            # Store results
            results.append({
                'epoch': epoch_index,
                'train_loss': training_loss,
                'train_acc': training_accuracy,
                'test_loss': testing_loss,
                'test_acc': testing_accuracy
            })

            # Update progress bar with current metrics
            pbar.set_postfix({
                'train_loss': f'{training_loss:.4f}',
                'train_acc': f'{training_accuracy*100:.1f}%',
                'test_loss': f'{testing_loss:.4f}',
                'test_acc': f'{testing_accuracy*100:.1f}%'
            })

            if epoch_index % self.snapshot_interval == 0:
                self.parameter_snapshots.append(Utils.flatten_params(
                    self.model.parameters()
                ))

            # Check for early stopping
            if self._should_early_stop(testing_loss):
                pbar.write(
                    f"Early stopping at epoch {epoch_index}/{self.maximum_epochs} - No improvement for {self.early_stop_patience} epochs")
                break

        # Print summary of all epochs at the end if needed
        if self.maximum_epochs > 0:
            last = results[-1]
            print(f"Final: train loss {last['train_loss']:.4f} acc {last['train_acc']*100:5.1f}% | "
                  f"test loss {last['test_loss']:.4f} acc {last['test_acc']*100:5.1f}%")

    def _should_early_stop(self, current_val_loss: float) -> bool:
        """Check if training should be stopped early due to lack of improvement."""
        if (self.best_val_loss - current_val_loss) > self.early_stop_min_delta:
            # We have improvement
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            return False
        else:
            # No significant improvement
            self.patience_counter += 1
            return self.patience_counter >= self.early_stop_patience

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

    def _save_snapshots(self, metrics: dict) -> None:
        optimizer_name = self.configuration.optimizer_choice.name
        target_directory = pathlib.Path("runs") / optimizer_name
        target_directory.mkdir(parents=True, exist_ok=True)
        # Save parameter snapshots
        with open(target_directory / "snapshots.pkl", "wb") as handle:
            pickle.dump(self.parameter_snapshots, handle)
        # Save runtime and memory metrics
        with open(target_directory / "metrics.json", "w") as mfile:
            json.dump(metrics, mfile)

        # Pre-compute loss surface data for later analysis (only if enabled)
        if self.configuration.hyper_parameters.precompute_surface and len(self.parameter_snapshots) > 1:
            self._pre_compute_loss_surface(target_directory)

        print(f"Snapshots and metrics saved to {target_directory}")

    # ------------------------------------------------------------------ #
    #                   LOSS SURFACE PRE-COMPUTATION                      #
    # ------------------------------------------------------------------ #

    def _pre_compute_loss_surface(self, output_dir: pathlib.Path) -> None:
        """Pre-compute loss surface values in PCA space to accelerate later analysis."""
        from sklearn.decomposition import PCA

        print("Pre-computing loss surface data for later analysis...")

        # Convert snapshots to numpy array for PCA
        snapshot_matrix = np.vstack(self.parameter_snapshots)

        # --- PCA basis ------------------------------------------------------------
        pca = PCA(n_components=2)
        projection = pca.fit_transform(snapshot_matrix)
        alpha_vals, beta_vals = projection.T
        dir1, dir2 = pca.components_
        theta_center = pca.mean_

        # --- grid extents slightly beyond trajectory ------------------------------
        padding = 0.5
        grid_points = 35  # Resolution of grid
        alpha_min, alpha_max = alpha_vals.min() - padding, alpha_vals.max() + padding
        beta_min, beta_max = beta_vals.min() - padding, beta_vals.max() + padding
        alpha_grid = np.linspace(alpha_min, alpha_max, grid_points)
        beta_grid = np.linspace(beta_min, beta_max, grid_points)
        loss_matrix = np.zeros((grid_points, grid_points))

        # --- Prepare for batch processing ---------------------------------------
        # Create a larger batch size for better GPU utilization
        large_batch_size = 4096

        # Collect all data into memory for faster processing
        all_features = []
        all_labels = []

        for features, labels in self.training_data_loader:
            all_features.append(features)
            all_labels.append(labels)

        # Concatenate all batches into one large tensor for more efficient processing
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)

        # Number of samples
        num_samples = all_features.shape[0]

        # Use the existing model instance and ensure it's in eval mode
        self.model.eval()

        # --- Process grid points in parallel batches -----------------------------
        surface_pbar = tqdm(range(0, grid_points),
                            desc="Computing loss surface", leave=False)

        # Create a flattened grid of all (alpha, beta) combinations for parallel processing
        all_alphas = []
        all_betas = []
        all_indices = []

        for i in range(grid_points):
            for j in range(grid_points):
                all_alphas.append(alpha_grid[i])
                all_betas.append(beta_grid[j])
                # Store indices for rebuilding the loss matrix
                all_indices.append((j, i))

        # Process in batches to avoid memory issues
        batch_size = 25  # Number of grid points to process at once
        for batch_start in range(0, len(all_alphas), batch_size):
            batch_end = min(batch_start + batch_size, len(all_alphas))

            # Get the current batch of grid points
            batch_alphas = all_alphas[batch_start:batch_end]
            batch_betas = all_betas[batch_start:batch_end]
            batch_indices = all_indices[batch_start:batch_end]

            # Update progress bar
            batch_midpoint = (batch_start + batch_end) // 2
            percent_complete = (batch_midpoint / len(all_alphas)) * 100
            surface_pbar.set_description(
                f"Computing surface: {percent_complete:.1f}% complete")
            surface_pbar.update(len(batch_indices) // grid_points)

            # For each grid point in this batch
            for k in range(len(batch_alphas)):
                alpha = batch_alphas[k]
                beta = batch_betas[k]
                j, i = batch_indices[k]

                # Compute parameters for this grid point
                theta = theta_center + alpha * dir1 + beta * dir2
                Utils.assign_flat_to_params(theta, self.model.parameters())

                # Process data in large batches for better GPU utilization
                total_loss = 0.0
                with torch.no_grad():
                    # Process the dataset in chunks that fit in GPU memory
                    for start_idx in range(0, num_samples, large_batch_size):
                        end_idx = min(
                            start_idx + large_batch_size, num_samples)

                        # Get batch
                        batch_x = all_features[start_idx:end_idx]
                        batch_y = all_labels[start_idx:end_idx]

                        # Forward pass
                        outputs = self.model(batch_x)
                        batch_loss = self.loss_function(outputs, batch_y)

                        # Accumulate loss
                        total_loss += batch_loss.item() * (end_idx - start_idx)

                # Store average loss for this grid point
                loss_matrix[j, i] = total_loss / num_samples

        surface_pbar.close()

        # Save the pre-computed surface data
        surface_data = {
            'alpha_grid': alpha_grid,
            'beta_grid': beta_grid,
            'loss_matrix': loss_matrix,
            'alpha_vals': alpha_vals,
            'beta_vals': beta_vals,
            'pca_components': pca.components_,
            'pca_mean': pca.mean_
        }

        with open(output_dir / "loss_surface_data.pkl", "wb") as handle:
            pickle.dump(surface_data, handle)

        print(
            f"Loss surface data saved to {output_dir / 'loss_surface_data.pkl'}")
