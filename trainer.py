from __future__ import annotations
import random
import numpy as np
import torch
from config import ExperimentConfig, ModelName
from data import load_dataset
from models import build_model
from losses import get_loss
from optimizers import get_optimizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from flatten import flatten_params
import pickle
import time
import pathlib


class Trainer:
    def __init__(self, configuration: ExperimentConfig):
        # reproducibility
        random.seed(configuration.hyperParameters.seed)
        np.random.seed(configuration.hyperParameters.seed)
        torch.manual_seed(configuration.hyperParameters.seed)

        features, labels = load_dataset(
            configuration.dataset, limit=None)   # full MNIST by default
        self.training_features, self.testing_features, self.training_labels, self.testing_labels = train_test_split(
            features, labels, test_size=0.2, random_state=configuration.hyperParameters.seed, stratify=labels)

        self.model = build_model(configuration.model)
        self.loss_fn = get_loss(configuration.loss)
        self.optimizer = get_optimizer(configuration.optimizer,
                                       self.model.parameters(),
                                       configuration.hyperParameters.learning_rate)
        self.epochs = configuration.hyperParameters.epochs
        self.batch_size = configuration.hyperParameters.batch_size

        # build PyTorch DataLoaders
        training_dataset = TensorDataset(torch.from_numpy(self.training_features),
                                         torch.from_numpy(self.training_labels))
        testing_dataset = TensorDataset(torch.from_numpy(self.testing_features),
                                        torch.from_numpy(self.testing_labels))
        self.training_data_loader = DataLoader(training_dataset, batch_size=self.batch_size,
                                               shuffle=True)
        self.testing_data_loader = DataLoader(testing_dataset, batch_size=1024)

        self.snapshots: list[np.ndarray] = []
        # default = each epoch
        self.snapshot_every = configuration.hyperParameters.snapshot_every or 1

    def _epoch(self, epoch_index: int):
        self.model.train()
        total_samples_processed, correct_predictions_count, current_epoch_loss = 0, 0, 0.0
        for feature_batch, label_batch in self.training_data_loader:
            self.optimizer.zero_grad()
            raw_model_outputs = self.model(feature_batch)
            loss_value = self.loss_fn(raw_model_outputs, label_batch)
            loss_value.backward()
            self.optimizer.step()

            current_epoch_loss += loss_value.item() * feature_batch.size(0)
            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                batch_predictions = raw_model_outputs.argmax(dim=1)
                correct_predictions_count += (batch_predictions ==
                                              label_batch).sum().item()
            total_samples_processed += feature_batch.size(0)

        # -------- snapshot logic ------------
        if epoch_index % self.snapshot_every == 0:
            self.snapshots.append(flatten_params(self.model.parameters()))
        # ------------------------------------

        average_epoch_loss = current_epoch_loss / total_samples_processed
        epoch_accuracy = correct_predictions_count / total_samples_processed
        return average_epoch_loss, epoch_accuracy

    def _eval(self):
        self.model.eval()
        total_samples_processed, correct_predictions_count, current_eval_loss = 0, 0, 0.0
        with torch.no_grad():
            for feature_batch, label_batch in self.testing_data_loader:
                raw_model_outputs = self.model(feature_batch)
                loss_value = self.loss_fn(raw_model_outputs, label_batch)
                current_eval_loss += loss_value.item() * feature_batch.size(0)
                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    batch_predictions = raw_model_outputs.argmax(dim=1)
                    correct_predictions_count += (batch_predictions ==
                                                  label_batch).sum().item()
                total_samples_processed += feature_batch.size(0)

        average_eval_loss = current_eval_loss / total_samples_processed
        eval_accuracy = correct_predictions_count / total_samples_processed
        return average_eval_loss, eval_accuracy

    # ------------------------------------------------------------------
    def fit(self):
        for epoch_number in range(1, self.epochs + 1):
            training_loss, training_accuracy = self._epoch(epoch_number)
            testing_loss,  testing_accuracy = self._eval()
            print(f"Epoch {epoch_number:02d} | "
                  f"Train Loss {training_loss:.4f} Acc {training_accuracy*100:5.1f}%  ||  "
                  f"Test Loss {testing_loss:.4f} Acc {testing_accuracy*100:5.1f}%")

        run_directory = pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")
        run_directory.mkdir(parents=True, exist_ok=True)
        with open(run_directory / "snapshots.pkl", "wb") as snapshot_file:
            pickle.dump(self.snapshots, snapshot_file)
