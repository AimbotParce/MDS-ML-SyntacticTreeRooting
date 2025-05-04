from abc import ABC
from typing import Callable, List, Literal, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainingLoop:
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        train_loader: DataLoader,
        epochs: int,
        callbacks: Optional[List["Callback"]] = [],
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        verbose: Literal[0, 1, 2] = 1,
    ):
        """
        Args:
            model (Module): The model to train.
            optimizer (Optimizer): The optimizer to use.
            loss_fn (Callable): The loss function to use.
            train_loader (DataLoader): The training data loader.
            val_loader (Optional[DataLoader], optional): The validation data loader. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.callbacks = callbacks
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0
        self.val_loss = 0
        self.loss = 0
        self.verbose = verbose
        self._stop_training = False

    def on_train_start(self):
        self.epoch = 0
        self.val_loss = 0
        self.loss = 0
        self._stop_training = False
        for callback in self.callbacks:
            callback.model = self.model
            callback.training_loop = self
            callback.on_train_start()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_start(self):
        self.metrics = {}
        self.avg_loss = 0
        for callback in self.callbacks:
            callback.on_epoch_start()

    def on_epoch_end(self):
        self.avg_loss /= len(self.train_loader)
        self.metrics["loss"] = self.avg_loss
        for callback in self.callbacks:
            callback.on_epoch_end()
        if self.verbose in [1, 2]:
            print(f"Epoch {self.epoch}/{self.epochs}: {self.formatted_metrics()}")

    def on_batch_start(self):
        for callback in self.callbacks:
            callback.on_batch_start()

    def on_batch_end(self):
        self.metrics["loss"] = self.loss
        self.avg_loss += self.loss
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_validation_start(self):
        for callback in self.callbacks:
            callback.on_validation_start()

    def on_validation_end(self):
        self.metrics["val_loss"] = self.val_loss
        for callback in self.callbacks:
            callback.on_validation_end()

    def on_validation_batch_start(self):
        for callback in self.callbacks:
            callback.on_validation_batch_start()

    def on_validation_batch_end(self):
        for callback in self.callbacks:
            callback.on_validation_batch_end()

    def stop_training(self):
        self._stop_training = True

    def formatted_metrics(self) -> str:
        """
        Format the metrics for display.
        Args:
            metrics (dict): The metrics to format.
        Returns:
            str: The formatted metrics.
        """
        return " ".join([f"{k}={v:.5f}" for k, v in self.metrics.items()])

    def train(self, verbose: int = 1):
        self.model.train()
        self.on_train_start()
        for self.epoch in range(1, self.epochs + 1):
            self.on_epoch_start()
            with tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {self.epoch}/{self.epochs}",
                leave=False,
                ncols=100,
                unit="batch",
                position=0,
                colour="green",
                disable=verbose != 1,
            ) as bar:
                for batch in self.train_loader:
                    self.on_batch_start()
                    out = self.model(batch.x, batch.edge_index)
                    loss = self.loss_fn(out, batch.y)
                    self.loss = loss.item()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.on_batch_end()
                    bar.set_postfix_str(self.formatted_metrics())
                    bar.update(1)

                val_loss = 0
                self.on_validation_start()
                for batch in self.val_loader:
                    with torch.no_grad():
                        self.on_validation_batch_start()
                        out = self.model(batch.x, batch.edge_index)
                        val_loss += self.loss_fn(out, batch.y).item()
                        self.on_validation_batch_end()
                self.val_loss = val_loss / len(self.val_loader)
                self.on_validation_end()
                bar.close()
                self.on_epoch_end()
                if self._stop_training:
                    break

        self.on_train_end()
        self._stop_training = False


from .callbacks import Callback
