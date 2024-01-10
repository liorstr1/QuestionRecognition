from datetime import datetime
from torch.nn.functional import softmax
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
import pickle

training_args = TrainingArguments(
        output_dir='./results',        # Output directory for model checkpoints
        num_train_epochs=10,            # Number of training epochs
        per_device_train_batch_size=2,# Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        warmup_steps=100,              # Number of warmup steps
        weight_decay=0.01,             # Weight decay for optimization
        logging_dir='./logs',          # Directory for storing logs
        logging_steps=100,              # Log every x updates steps
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # fp16=True
    )


def eval(output):
    predictions = np.argmax(output.predictions, axis=-1)
    # con = confusion_matrix(output.label_ids, predictions, labels=[0, 1, 2, 3, 4])
    # print(con)
    return {"f1": f1_score(output.label_ids, predictions, average='weighted')}


def train_and_evaluate_transformers(
    split,
    model_full_name,

):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))

    X_train, y_train = split['train']
    X_test, y_test, _ = split['test']

    tokenizer = AutoTokenizer.from_pretrained(model_full_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_full_name, num_labels=len(set(y_train)))

    train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt")
    train_dataset = Dataset(train_encodings, y_train)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, return_tensors="pt")
    test_dataset = Dataset(test_encodings, y_test)
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=eval,
    )
    trainer.train()

    test_trainer_predictions = trainer.predict(test_dataset)
    test_trainer_predictions = np.argmax(
        torch.softmax(torch.tensor(test_trainer_predictions.predictions), dim=-1).numpy(), axis=1
    )
    return f1_score(y_test, test_trainer_predictions, average='weighted')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def cross_validate_and_save_model(model_name, model_full_name, docs, labels, saved_model_path, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    X, y = np.array(docs), np.array(labels)
    f1_scores = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    print(f"{datetime.now().strftime('%H:%M:%S')}:start training")
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"{datetime.now().strftime('%H:%M:%S')}:Training on fold {fold + 1}/{n_splits}...")

        # Splitting the data
        X_train, X_val = X[train_index].tolist(), X[val_index].tolist()
        y_train, y_val = y[train_index], y[val_index]

        # Tokenization and dataset creation
        tokenizer = AutoTokenizer.from_pretrained(model_full_name)
        train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt")
        val_encodings = tokenizer(X_val, truncation=True, padding=True, return_tensors="pt")
        train_dataset = DatasetModel(train_encodings, y_train)
        val_dataset = DatasetModel(val_encodings, y_val)

        # Model training
        model = AutoModelForSequenceClassification.from_pretrained(model_full_name, num_labels=len(set(y_train)))
        model.to(device)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=eval,
        )
        trainer.train()

        # Evaluate and store F1 score
        val_preds = trainer.predict(val_dataset)
        val_f1 = f1_score(y_val, np.argmax(val_preds.predictions, axis=-1), average='weighted')
        f1_scores.append(val_f1)

        # Save the model after the last fold
        if fold == n_splits - 1:
            model_path = f'{saved_model_path}/{model_name}.bin'
            pickle.dump(model, open(model_path, 'wb'))
            print(f"{datetime.now().strftime('%H:%M:%S')}:Model saved to {model_path}")

    return np.mean(f1_scores)


def predict_with_confidence(model_name, model_full_name, docs, saved_model_path):
    # Load the model from the saved path
    model_path = f'{saved_model_path}/{model_name}.bin'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Ensure the model is in evaluation mode
    model.eval()

    # Tokenize the documents
    tokenizer = AutoTokenizer.from_pretrained(model_full_name)
    encodings = tokenizer(docs, truncation=True, padding=True, return_tensors="pt")

    # Move to the same device as the model
    device = next(model.parameters()).device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    # Make predictions without gradient calculation
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Apply softmax to get confidence scores for all labels
    probabilities = softmax(outputs.logits, dim=1).cpu().numpy()

    # List to store dictionaries of confidence scores for each document
    all_confidence_scores = []

    # Iterate over each document's probabilities and create a dictionary
    for doc_probs in probabilities:
        # Create a dictionary for the current document
        doc_scores = {label_idx: score for label_idx, score in enumerate(doc_probs)}
        all_confidence_scores.append(doc_scores)

    # Get the highest confidence score and the corresponding prediction
    confidence_scores = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)

    return predictions, confidence_scores, all_confidence_scores


class DatasetModel(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are long integers
        return item

    def __len__(self):
        return len(self.labels)
