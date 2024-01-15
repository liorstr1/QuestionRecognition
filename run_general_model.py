from datetime import datetime
from torch.nn.functional import softmax
import numpy as np
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments,
    EvalPrediction
)
from torch.optim import AdamW
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch.nn.functional as F

# Distillation parameters
temperature = 2.0  # Can be tuned


training_args = TrainingArguments(
        output_dir='./results',        # Output directory for model checkpoints
        num_train_epochs=10,            # Number of training epochs
        per_device_train_batch_size=2, # Batch size for training
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


def predict_with_confidence(model_path, model_full_name, docs, model):
    # Load the model from the saved path
    if model is None:
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


def cross_validate_and_save_student_model(docs, labels, teacher_model_path, student_model_path, n_splits=5):
    with open(teacher_model_path, 'rb') as file:
        teacher_model = pickle.load(file)
    teacher_model.eval()

    student_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(labels)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    X, y = np.array(docs), np.array(labels)
    f1_scores = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model.to(device)
    teacher_model.to(device)

    local_training_args = TrainingArguments(
        output_dir='./results',         # Output directory for model checkpoints
        num_train_epochs=3,             # Number of training epochs
        per_device_train_batch_size=16, # Batch size for training
        per_device_eval_batch_size=32,  # Batch size for evaluation
        warmup_steps=500,               # Number of warmup steps
        weight_decay=0.01,              # Weight decay for optimization
        logging_dir='./logs',           # Directory for storing logs
        logging_steps=10,               # Log every x updates steps
        evaluation_strategy="steps"
    )

    optimizer = AdamW(student_model.parameters(), lr=5e-5)

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index].tolist(), X[val_index].tolist()
        y_train, y_val = y[train_index], y[val_index]

        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt")
        val_encodings = tokenizer(X_val, truncation=True, padding=True, return_tensors="pt")

        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train))
        val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(y_val))

        for epoch in range(local_training_args.num_train_epochs):
            student_model.train()
            for batch in DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
                    teacher_predictions = F.softmax(teacher_outputs.logits / temperature, dim=-1)

                # Student forward pass
                student_outputs = student_model(input_ids, attention_mask=attention_mask)
                student_predictions = student_outputs.logits

                # Calculate distillation loss
                distillation_loss = F.kl_div(
                    F.log_softmax(student_predictions / temperature, dim=-1),
                    teacher_predictions,
                    reduction='batchmean'
                ) * (temperature ** 2)

                # Calculate standard classification loss
                labels = labels.long()
                classification_loss = F.cross_entropy(student_predictions, labels)

                # Combine losses
                loss = distillation_loss + classification_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Evaluate model after each epoch
            student_model.eval()
            eval_predictions = []
            eval_labels = []
            for batch in DataLoader(val_dataset, batch_size=local_training_args.per_device_eval_batch_size):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                with torch.no_grad():
                    outputs = student_model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    eval_predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    eval_labels.extend(labels.cpu().numpy())

            # Compute F1 score
            if eval_predictions and eval_labels:
                f1 = f1_score(eval_labels, eval_predictions, average='weighted')
                f1_scores.append(f1)
                print(f"Fold {fold + 1}, Epoch {epoch + 1}: F1 Score - {f1}")

        # Save model after the last fold
        if fold == n_splits - 1:
            student_model.save_pretrained(student_model_path)
            tokenizer.save_pretrained(student_model_path)

    return np.mean(f1_scores)
