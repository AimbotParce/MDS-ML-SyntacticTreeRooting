from abc import ABC

from torch.nn import Module

from .training import TrainingLoop


class Callback(ABC):
    training_loop: TrainingLoop
    model: Module

    def on_epoch_end(self):
        pass

    def on_epoch_start(self):
        pass

    def on_batch_end(self):
        pass

    def on_batch_start(self):
        pass

    def on_train_end(self):
        pass

    def on_train_start(self):
        pass

    def on_validation_start(self):
        pass

    def on_validation_end(self):
        pass

    def on_validation_batch_end(self):
        pass

    def on_validation_batch_start(self):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def on_validation_end(self):
        current_score = self.training_loop.val_loss
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.counter} epochs without improvement.")
                self.training_loop.stop_training()


class ReduceLROnPlateau(Callback):
    def __init__(self, factor=0.1, patience=5):
        self.factor = factor
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def on_epoch_end(self):
        # First store the current learning rate in the metrics
        for j, param_group in enumerate(self.training_loop.optimizer.param_groups, 1):
            self.training_loop.metrics[f"lr_{j}"] = param_group["lr"]

        current_score = self.training_loop.val_loss
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Reducing learning rate by a factor of {self.factor}.")
                for param_group in self.training_loop.optimizer.param_groups:
                    param_group["lr"] *= self.factor
                self.counter = 0
