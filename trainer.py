from __future__ import annotations

import pathlib
import pickle
import random
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from data import load_dataset
from utils import Utils
from losses import get_loss
from models import build_model
from optimizers import get_optimizer


class Trainer:
    """
    Handles the training and evaluation of a neural network model based on a given configuration.
    """

    def __init__(self, configuration: ExperimentConfig):
        """
        Initializes the Trainer with the given experiment configuration.

        Args:
            configuration: The configuration object for the experiment.
        """
        self.config = configuration
        self.computation_device = Utils.choose_device(
            self.config.hyperParameters.use_cuda)
        print(">>> Training on", self.computation_device)
        self._set_seeds()
        self._prepare_data()
        self._initialize_training_components()
        self._create_dataloaders()

        self.snapshots: list[np.ndarray] = []
        # default = each epoch
        self.snapshot_every = self.config.hyperParameters.snapshot_every or 1

    def _set_seeds(self) -> None:
        """Sets random seeds for reproducibility."""
        seed = self.config.hyperParameters.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _prepare_data(self) -> None:
        """Loads and splits the dataset into training and testing sets."""
        """Featuresa are the input data, and labels are the target values."""
        features, labels = load_dataset(
            self.config.dataset, limit=None
        )  # full MNIST by default
        self.training_features, self.testing_features, self.training_labels, self.testing_labels = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=self.config.hyperParameters.seed,
            stratify=labels,
        )

    def _initialize_training_components(self) -> None:
        """Initializes the model, loss function, and optimizer."""
        self.model = build_model(self.config.model).to(self.computation_device)
        self.loss_function = get_loss(
            self.config.loss).to(self.computation_device)

        self.optimizer = get_optimizer(
            choice=self.config.optimizer_choice,
            model_parameters=self.model.parameters(),
            learning_rate=self.config.hyper_parameters.learning_rate,
        )

        self.epochs = self.config.hyperParameters.epochs
        self.batch_size = self.config.hyperParameters.batch_size

    def _create_dataloaders(self) -> None:
        """
            Creates PyTorch DataLoaders for training and testing
            Converts the NumPy feature and label arrays into PyTorch tensors that
            already live on self.computation_device, then wraps them in DataLoaders.
        """
        # --- convert training split ------------------------------------------------
        training_feature_tensor = torch.tensor(
            self.training_features, dtype=torch.float32,
            device=self.computation_device
        )
        training_label_tensor = torch.tensor(
            self.training_labels, dtype=torch.long,
            device=self.computation_device
        )

        # --- convert testing split -------------------------------------------------
        testing_feature_tensor = torch.tensor(
            self.testing_features, dtype=torch.float32,
            device=self.computation_device
        )
        testing_label_tensor = torch.tensor(
            self.testing_labels, dtype=torch.long,
            device=self.computation_device
        )

        # --- wrap tensors in TensorDataset objects ---------------------------------
        training_dataset = TensorDataset(training_feature_tensor,
                                         training_label_tensor)
        testing_dataset = TensorDataset(testing_feature_tensor,
                                        testing_label_tensor)

        # --- finally build the DataLoaders -----------------------------------------
        self.training_data_loader = DataLoader(
            dataset=training_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.testing_data_loader = DataLoader(
            dataset=testing_dataset,
            batch_size=4_096          # single large batch for evaluation
        )

    def _epoch(self, epoch_index: int) -> tuple[float, float]:
        """
        Performs a single training epoch.

        Args:
            epoch_index: The current epoch number.

        Returns:
            A tuple containing the average training loss and accuracy for the epoch.
        """
        self.model.train()
        total_samples_processed, correct_predictions_count, current_epoch_loss = (
            0,
            0,
            0.0,
        )
        for feature_batch, label_batch in self.training_data_loader:
            self.optimizer.zero_grad()
            raw_model_outputs = self.model(feature_batch)
            loss_value = self.loss_fn(raw_model_outputs, label_batch)
            loss_value.backward()
            self.optimizer.step()

            current_epoch_loss += loss_value.item() * feature_batch.size(0)
            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                batch_predictions = raw_model_outputs.argmax(dim=1)
                correct_predictions_count += (
                    (batch_predictions == label_batch).sum().item()
                )
            total_samples_processed += feature_batch.size(0)

        if epoch_index % self.snapshot_every == 0:
            self.snapshots.append(
                Utils.flatten_params(self.model.parameters()))

        average_epoch_loss = current_epoch_loss / total_samples_processed
        epoch_accuracy = correct_predictions_count / total_samples_processed
        return average_epoch_loss, epoch_accuracy

    def _eval(self) -> tuple[float, float]:
        """
        Evaluates the model on the testing dataset.

        Returns:
            A tuple containing the average evaluation loss and accuracy.
        """
        self.model.eval()
        total_samples_processed, correct_predictions_count, current_eval_loss = (
            0,
            0,
            0.0,
        )
        with torch.no_grad():
            for feature_batch, label_batch in self.testing_data_loader:
                raw_model_outputs = self.model(feature_batch)
                loss_value = self.loss_fn(raw_model_outputs, label_batch)
                current_eval_loss += loss_value.item() * feature_batch.size(0)
                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    batch_predictions = raw_model_outputs.argmax(dim=1)
                    correct_predictions_count += (
                        (batch_predictions == label_batch).sum().item()
                    )
                total_samples_processed += feature_batch.size(0)

        average_eval_loss = current_eval_loss / total_samples_processed
        eval_accuracy = correct_predictions_count / total_samples_processed
        return average_eval_loss, eval_accuracy

    def fit(self) -> None:
        """
        Trains the model for the configured number of epochs and saves parameter snapshots.
        """
        print(
            f"Starting training for {self.config.model.value} model "
            f"with {self.config.optimizer.value} optimizer "
            f"for {self.epochs} epochs."
        )
        for epoch_number in range(1, self.epochs + 1):
            training_loss, training_accuracy = self._epoch(epoch_number)
            testing_loss, testing_accuracy = self._eval()
            print(
                f"Epoch {epoch_number:02d} | "
                f"Train Loss {training_loss:.4f} Acc {training_accuracy*100:5.1f}%  ||  "
                f"Test Loss {testing_loss:.4f} Acc {testing_accuracy*100:5.1f}%"
            )

        self._save_snapshots()
        print("Training complete. Snapshots saved.")

    def _save_snapshots(self) -> None:
        """Saves the collected parameter snapshots to a pickle file."""
        run_directory = pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")
        run_directory.mkdir(parents=True, exist_ok=True)
        snapshot_file_path = run_directory / "snapshots.pkl"
        with open(snapshot_file_path, "wb") as snapshot_file:
            pickle.dump(self.snapshots, snapshot_file)
        print(f"Snapshots saved to {snapshot_file_path}")
