import itertools as it
import json
from pathlib import Path
from typing import Dict, Generic, Iterator, List, TypeVar

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1_l2

T = TypeVar("T")


def create_model(
    input_shape,
    hidden_layer_sizes: List[int],
    first_layer_l1: float = 0.0,
    first_layer_l2: float = 0.0,
    hidden_layer_l1: float = 0.0,
    hidden_layer_l2: float = 0.0,
    first_layer_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    initial_learning_rate: float = 0.001,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-08,
) -> Model:
    """
    Create a simple feedforward neural network model.
    """
    if len(hidden_layer_sizes) == 0:
        raise ValueError("hidden_layer_sizes must contain at least one layer size.")

    inputs = Input(shape=(input_shape,))
    x = Dense(
        hidden_layer_sizes[0],
        activation="relu",
        kernel_regularizer=l1_l2(first_layer_l1, first_layer_l2),
    )(inputs)
    if first_layer_dropout > 0:
        x = Dropout(first_layer_dropout)(x)

    for layer_size in hidden_layer_sizes[1:]:
        x = Dense(layer_size, activation="relu", kernel_regularizer=l1_l2(hidden_layer_l1, hidden_layer_l2))(x)
        if hidden_dropout > 0:
            x = Dropout(hidden_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=initial_learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'language' column in the DataFrame.
    """
    return pd.get_dummies(df, columns=["language"], prefix="", prefix_sep="", drop_first=False)


class Dimension(Generic[T]):
    def __init__(self, *values: T):
        self._values = values

    @property
    def options(self):
        return self._values

    def __len__(self):
        return len(self._values)


class GridSearch:
    """
    Perform a grid search over a set of dimensions. Dimensions will be iterated over in the revere order they were added.
    """

    def __init__(self, dimensions: Dict[str, Dimension] = {}):
        self._dimensions = dimensions

    def add_dimension(self, key: str, dimension: Dimension):
        self._dimensions[key] = dimension

    def __len__(self):
        return np.prod(list(map(lambda x: len(x), self._dimensions.values())))

    def __iter__(self) -> Iterator[str]:
        for prod in it.product(*map(lambda x: x.options, self._dimensions.values())):
            yield {key: value for key, value in zip(self._dimensions.keys(), prod)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MLP model training with grid search.")
    parser.add_argument("--skip-configurations", type=int, default=None, help="Number of configurations to skip.")
    parser.add_argument(
        "--no-run-increment", action="store_true", help="Do not increment the run ID, use 'run-1' only."
    )
    args = parser.parse_args()
    skip_configurations: int = args.skip_configurations
    no_run_increment: bool = args.no_run_increment

    # Load the data
    train_data = pd.read_csv("../data/cache/training_unwound.csv")
    validation_data = pd.read_csv("../data/cache/validation_unwound.csv")
    train_data["language"] = train_data["language"].astype("category")
    validation_data["language"] = validation_data["language"].astype("category")
    train_data.head()

    grid = GridSearch()
    grid.add_dimension("hidden_layer_sizes", Dimension([64, 64, 32], [64, 32], [64, 32, 16, 8]))
    grid.add_dimension("first_layer_l1", Dimension(0.0, 0.01, 0.1))
    grid.add_dimension("first_layer_l2", Dimension(0.0, 0.01))
    grid.add_dimension("hidden_layer_l1", Dimension(0.0, 0.01))
    grid.add_dimension("hidden_layer_l2", Dimension(0.0, 0.01, 0.1))
    grid.add_dimension("first_layer_dropout", Dimension(0.2))
    grid.add_dimension("hidden_dropout", Dimension(0.0, 0.1, 0.2))
    grid.add_dimension("initial_learning_rate", Dimension(0.001))
    grid.add_dimension("beta_1", Dimension(0.9, 0.95, 0.99))
    grid.add_dimension("beta_2", Dimension(0.999, 0.995, 0.99))
    grid.add_dimension("epsilon", Dimension(1e-08, 1e-07, 1e-06))

    print(f"Total configurations: {len(grid)}")

    cross_validation_folds = 5

    models_path_root = Path("../data/models/mlp")

    models_path = models_path_root / "run-1"
    if not no_run_increment:
        run_id = 1
        while models_path.exists():
            run_id += 1
            models_path = models_path_root / f"run-{run_id}"
    models_path.mkdir(parents=True, exist_ok=True)

    row_indices = train_data["row_index"].unique()
    # We'll separate based on row indices, because that's what we have now. Ideally
    # we would separate based on sentence id, but we don't have that in the data now
    for j, configuration in enumerate(grid):
        if skip_configurations is not None and j <= skip_configurations:
            print(f"Skipping configuration {j + 1}/{len(grid)}")
            continue
        print(f"Configuration {j + 1}/{len(grid)}: {configuration}")
        config_path = models_path / f"configuration-{j}"
        config_path.mkdir(parents=True, exist_ok=True)
        with open(config_path / "configuration.json", "w") as f:
            json.dump(configuration, f, indent=4)

        losses = []
        accuracies = []

        # Separate cross-validation data
        for i in range(cross_validation_folds):
            fold_path = config_path / f"fold-{i}"
            fold_path.mkdir(parents=True, exist_ok=True)

            fold_row_indices = row_indices[i::cross_validation_folds]
            with open(fold_path / "fold_validation_row_indices.json", "w") as f:
                json.dump(fold_row_indices.tolist(), f)

            train_fold_data = train_data[~train_data["row_index"].isin(fold_row_indices)]
            validation_fold_data = train_data[train_data["row_index"].isin(fold_row_indices)]

            # One-hot encode the training and validation data
            X_train = one_hot_encode(train_fold_data.drop(columns=["row_index", "node", "is_root"]))
            y_train = train_fold_data["is_root"]
            X_val = one_hot_encode(validation_fold_data.drop(columns=["row_index", "node", "is_root"]))
            y_val = validation_fold_data["is_root"]

            model = create_model(X_train.shape[1], **configuration)
            model.summary()

            class_weights = y_train.value_counts(normalize=True).to_dict()
            class_weights = {k: 1.0 / v for k, v in class_weights.items()}
            class_weights

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=str(fold_path / "best_model.keras"), monitor="val_loss", save_best_only=True),
            ]

            model.fit(
                X_train,
                y_train,
                epochs=10,
                batch_size=256,
                class_weight=class_weights,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                shuffle=True,
                verbose=2,
            )
            model.save(fold_path / "final_model.keras")
            loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
            losses.append(loss)
            accuracies.append(accuracy)
            with open(fold_path / "metrics.json", "w") as f:
                json.dump({"loss": loss, "accuracy": accuracy}, f, indent=4)
            with open(config_path / "metrics.json", "w") as f:  # Store this every fold, so we "checkpoint".
                json.dump({"losses": losses, "accuracies": accuracies}, f, indent=4)
            print(f"Fold {i + 1}/{cross_validation_folds} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print(
            f"Configuration {j + 1}/{len(grid)} - Average Loss: {sum(losses) / len(losses):.4f}, "
            f"Average Accuracy: {sum(accuracies) / len(accuracies):.4f}"
        )
        with open(config_path / "metrics.json", "w") as f:
            json.dump(
                {
                    "losses": losses,
                    "accuracies": accuracies,
                    "average_loss": sum(losses) / len(losses),
                    "average_accuracy": sum(accuracies) / len(accuracies),
                },
                f,
                indent=4,
            )
        print(f"Configuration {j + 1}/{len(grid)} completed and saved.")
