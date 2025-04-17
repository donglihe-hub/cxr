from pathlib import Path
from typing import Iterable

import tensorflow as tf
import tensorflow_models as tfm
from official.modeling.optimization import lars
import numpy as np
import tqdm
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]

        return embedding, label

def main():
    TOKEN_NUM = 32
    EMBEDDINGS_SIZE = 768

    train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels = get_dataset()
    # Prepare the training and validation datasets using embeddings and diagnosis labels
    training_data = create_tf_dataset_from_embeddings(
        embeddings=np.asarray(train_embeddings),
        labels=train_labels,
        embeddings_size=TOKEN_NUM * EMBEDDINGS_SIZE)

    validation_data = create_tf_dataset_from_embeddings(
        embeddings=np.asarray(val_embeddings),
        labels=val_labels,
        embeddings_size=TOKEN_NUM * EMBEDDINGS_SIZE)

    test_data = create_tf_dataset_from_embeddings(
        embeddings=np.asarray(test_embeddings),
        labels=test_labels,
        embeddings_size=TOKEN_NUM * EMBEDDINGS_SIZE)

    # Create the model with the specified configuration
    model = create_model(
        ["TB"],
        token_num=TOKEN_NUM,
        embeddings_size = EMBEDDINGS_SIZE,
        learning_rate=0.1,
        dropout=0.2,
    )
    # import pdb;pdb.set_trace()
    # Train the model using the prepared datasets, with specific batch sizes and caching strategies
    model.fit(
        x=training_data.batch(512).prefetch(tf.data.AUTOTUNE).cache(),
        validation_data=validation_data.batch(1).cache(),
        epochs=100,
    )
    results = model.evaluate(test_data.batch(1).cache())
    # print(results)

    # Display the model architecture summary
    # model.summary()

def get_dataset():
    data_dir = Path("./data/XR_DICOM")

    data_paths = sorted(data_dir.glob(f"**/*_general.npy"))
    embeddings = [np.load(i).astype(np.float32).ravel() for i in tqdm.tqdm(data_paths, desc="Loading embedding files")]
    labels = np.array([1 if "Abnormal" in str(i) else 0 for i in data_paths])
    _, counts = np.unique(labels, return_counts=True)
    print(
        f"Normal instances: {counts[0]}; Abnormal instances: {counts[1]}"
    )

    train_len = 0.7
    val_len = 0.1
    test_len = 0.2
    dataset = EmbeddingDataset(embeddings, labels)
    test_dataset, val_dataset, train_dataset = random_split(
        dataset=dataset,
        lengths=[test_len, val_len, train_len],
        generator=torch.Generator().manual_seed(42),
    )

    # import pdb; pdb.set_trace()
    train_embeddings = train_dataset.dataset.embeddings
    train_labels = train_dataset.dataset.labels.astype(np.float64)
    val_embeddings = val_dataset.dataset.embeddings
    val_labels = val_dataset.dataset.labels.astype(np.float64)
    test_embeddings = test_dataset.dataset.embeddings
    test_labels = test_dataset.dataset.labels.astype(np.float64)

    return train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels

def create_tf_dataset_from_embeddings(
    embeddings: Iterable[np.ndarray],
    labels: Iterable[int],
    embeddings_size: int
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from embeddings, image IDs, and labels."""
    # Ensure embeddings, image_ids, and labels are lists
    embeddings = list(embeddings)
    labels = list(labels)

    # Check that the lengths match
    assert len(embeddings) == len(labels), \
        "Lengths of embeddings, and labels must be equal"

    # Convert embeddings to np.float32 if necessary
    embeddings = [np.asarray(e, dtype=np.float32) for e in embeddings]

    # Create datasets for embeddings and labels
    ds_embeddings = tf.data.Dataset.from_tensor_slices(embeddings)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)

    # Zip embeddings and labels into a single dataset
    dataset = tf.data.Dataset.zip((ds_embeddings, ds_labels))

    return dataset


def create_model(heads,
                 token_num,
                 embeddings_size,
                 learning_rate=0.1,
                 end_lr_factor=1.0,
                 dropout=0.0,
                 decay_steps=1000,
                 loss_weights=None,
                 hidden_layer_sizes=[512, 256],
                 weight_decay=0.0,
                 seed=None) -> tf.keras.Model:
    """
    Creates linear probe or multilayer perceptron using LARS + cosine decay.
    """
    inputs = tf.keras.Input(shape=(token_num * embeddings_size,))
    inputs_reshape = tf.keras.layers.Reshape((token_num, embeddings_size))(inputs)
    inputs_pooled = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(inputs_reshape)
    hidden = inputs_pooled
    # If no hidden_layer_sizes are provided, model will be a linear probe.
    for size in hidden_layer_sizes:
        hidden = tf.keras.layers.Dense(
            size,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2=weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(l2=weight_decay))(
                hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Dropout(dropout, seed=seed)(hidden)
    output = tf.keras.layers.Dense(
        units=len(heads),
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(
            hidden)

    outputs = {}
    for i, head in enumerate(heads):
        outputs[head] = tf.keras.layers.Lambda(
            lambda x: x[..., i:i + 1], name=head.lower())(
                output)

    model = tf.keras.Model(inputs, outputs)
    learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
        tf.cast(learning_rate, tf.float32),
        tf.cast(decay_steps, tf.float32),
        alpha=tf.cast(end_lr_factor, tf.float32))
    model.compile(
        optimizer=tfm.optimization.lars.LARS(
            learning_rate=learning_rate_fn),
        loss=dict([(head, 'binary_crossentropy') for head in heads]),
        loss_weights=loss_weights or dict([(head, 1.) for head in heads]),
        weighted_metrics=[
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.F1Score(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()])
    return model

if __name__ == "__main__":
    main()