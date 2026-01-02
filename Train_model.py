import os
import numpy as np
import tensorflow as tf
from collections import Counter
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset


BASE_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
OUTPUT_DIR = "finetuned_sentiment"

TEXT_COLUMN = "conversation"
LABEL_COLUMN = "customer_sentiment"

NUM_CLASSES = 3


def map_label_to_id(label):
    label = str(label).strip().lower()
    if label in ["negative", "frustrated", "neg", "0"]:
        return 0
    if label in ["neutral", "neu", "1"]:
        return 1
    if label in ["positive", "pos", "2"]:
        return 2
    raise ValueError(f"Unknown label: {label}")


def build_tf_dataset(texts, labels, tokenizer, batch_size, shuffle):
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf",
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
            tf.convert_to_tensor(labels, dtype=tf.int32),
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(texts), 10_000),
            reshuffle_each_iteration=True,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(labels):
    counts = Counter(labels.tolist())
    total_samples = len(labels)
    return {
        class_id: total_samples / (NUM_CLASSES * max(counts.get(class_id, 0), 1))
        for class_id in range(NUM_CLASSES)
    }


def oversample_positives(texts, labels, target_positive_fraction=0.30, seed=42):
    rng = np.random.default_rng(seed)
    positive_indices = np.where(labels == 2)[0]

    if len(positive_indices) == 0:
        return texts, labels

    target_positive_count = int(target_positive_fraction * len(labels))
    additional_needed = target_positive_count - len(positive_indices)

    if additional_needed <= 0:
        return texts, labels

    sampled_indices = rng.choice(
        positive_indices,
        size=additional_needed,
        replace=True,
    )

    texts_extended = np.concatenate([texts, texts[sampled_indices]])
    labels_extended = np.concatenate([labels, labels[sampled_indices]])

    permutation = rng.permutation(len(labels_extended))
    return texts_extended[permutation], labels_extended[permutation]


def train_sentiment_model(
    texts,
    labels,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    validation_split,
):
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels, dtype=np.int32)

    if len(texts) < 20:
        raise ValueError("Dataset too small")

    permutation = np.random.permutation(len(texts))
    texts = texts[permutation]
    labels = labels[permutation]

    val_size = max(1, int(len(texts) * validation_split))

    texts_val = texts[:val_size]
    labels_val = labels[:val_size]

    texts_train = texts[val_size:]
    labels_train = labels[val_size:]

    texts_train, labels_train = oversample_positives(
        texts_train,
        labels_train,
        target_positive_fraction=0.30,
    )

    class_weights = compute_class_weights(labels_train)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = TFAutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME)

    train_dataset = build_tf_dataset(
        texts_train,
        labels_train,
        tokenizer,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataset = build_tf_dataset(
        texts_val,
        labels_val,
        tokenizer,
        batch_size=batch_size,
        shuffle=False,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        verbose=1,
    )

    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    model_dir = os.path.join(output_dir, "model")

    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    tokenizer.save_pretrained(tokenizer_dir)
    model.save_pretrained(model_dir)

    return history.history, tokenizer_dir, model_dir, class_weights


def load_training_data():
    dataset = load_dataset("NebulaByte/E-Commerce_Customer_Support_Conversations")
    train_split = dataset["train"]
    texts = [str(text) for text in train_split[TEXT_COLUMN]]
    labels = [map_label_to_id(label) for label in train_split[LABEL_COLUMN]]
    return texts, labels


if __name__ == "__main__":
    NUM_EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    VALIDATION_SPLIT = 0.10

    texts, labels = load_training_data()

    history, tokenizer_path, model_path, class_weights = train_sentiment_model(
        texts=texts,
        labels=labels,
        output_dir=OUTPUT_DIR,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        validation_split=VALIDATION_SPLIT,
    )

    print("Tokenizer saved to:", tokenizer_path)
    print("Model saved to:", model_path)
    print("Class weights:", class_weights)
    print("Final training loss:", history["loss"][-1])

    if "val_loss" in history:
        print("Final validation loss:", history["val_loss"][-1])
