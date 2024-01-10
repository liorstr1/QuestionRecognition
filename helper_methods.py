import csv

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def read_question_data(question_path):
    with open(question_path, 'r') as f:
        all_lines = f.readlines()
    return [li for li in all_lines if li.strip()]


def read_question_enrich_data(question_enrich_path):
    # Dictionary to store the data
    data_dict = {}
    # Open the CSV file and read it
    with open(question_enrich_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Check if the row is not empty
                key = row[0]  # First element as key
                value = row[1:]  # Rest of the elements as value
                data_dict[key] = value
    return data_dict


def save_dict_to_csv(data_dict, file_name):
    # Open the file for writing
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write each key-value pair to the file
        for key, values in data_dict.items():
            # Create a row with the key and the values
            row = [key] + values
            writer.writerow(row)


def split_to_train_and_test(docs, labels, test_size, n_splits=1):
    docs = np.array(docs)
    labels = np.array(labels)

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    splits = []

    # Generate the splits
    for train_index, test_index in sss.split(docs, labels):
        X_train, X_test = docs[train_index], docs[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        splits.append({
            'train': (X_train.tolist(), y_train.tolist(), train_index.tolist()),
            'test': (X_test.tolist(), y_test.tolist(), test_index.tolist())
        })
    return splits


def get_data(question_path):
    question_enrich_path = question_path[:-4] + ".csv"
    X = read_question_data(question_path)
    question_enrich_dict = read_question_enrich_data(question_enrich_path)
    return X, question_enrich_dict


def enrich_data(split_data, question_enrich_dict):
    new_questions, new_labels = [], []
    for question, label in zip(split_data[0], split_data[1]):
        new_questions.append(question)
        new_labels.append(label)
        new_questions.extend(question_enrich_dict[question])
        new_labels.extend([label] * len(question_enrich_dict[question]))
    return new_questions, new_labels


def get_docs_and_indices(heb_doc2docs):
    # Initialize the list to hold all values
    all_values = []

    # Initialize the dictionary for start and end indices
    indices_dict = {}

    # Current start index
    start_index = 0

    for key, values in heb_doc2docs.items():
        # Extend the all_values list with values from the current key
        all_values.extend(values)

        # The end index is the new length of all_values minus 1
        end_index = len(all_values) - 1

        # Assign the start and end indices to the key in indices_dict
        indices_dict[key] = (start_index, end_index)

        # Update the start_index for the next iteration
        start_index = end_index + 1

    return all_values, indices_dict


def calculate_results(indices_dict, all_confidence_scores, opt_labels):
    def calculate_prob(opt):
        return sum(d[opt] for d in conf_scores) / len(conf_scores)

    res = {}
    for heb_d, indices in indices_dict.items():
        conf_scores = all_confidence_scores[indices[0]: indices[1]]

        all_opt_conf = {opt: calculate_prob(opt) for opt in opt_labels}
        key, value = max(all_opt_conf.items(), key=lambda item: item[1])
        res[heb_d] = (key, value)
    return res


def get_label2answer(answer_path):
    with open(answer_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    # Remove duplicates while preserving order
    seen = set()
    all_lines_ordered = [x for x in all_lines if not (x in seen or seen.add(x))]

    # Remove empty lines and create a dictionary with indices
    all_lines_ordered = [li.strip() for li in all_lines_ordered if li.strip()]
    return {idx: l for idx, l in enumerate(all_lines_ordered)}

