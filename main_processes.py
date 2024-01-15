import os.path
from collections import defaultdict

from create_variants import CreateParaphrase
from hebrew_to_english import TranslateSentences
from helper_methods import (
    read_question_data, read_question_enrich_data, split_to_train_and_test, get_data, enrich_data, get_docs_and_indices,
    calculate_results
)
from run_general_model import (
    train_and_evaluate_transformers, cross_validate_and_save_model, predict_with_confidence,
    cross_validate_and_save_student_model
)

MODEL_NAME = 'stella'
MODEL_FULL_NAME = 'infgrad/stella-base-en-v2'


def enrichment_process(question_path):
    question_enrich_path = question_path[:-4] + ".csv"
    all_questions = read_question_data(question_path)
    question_enrich_dict = read_question_enrich_data(question_enrich_path)
    if len(question_enrich_dict) < len(all_questions):
        cp = CreateParaphrase()
        while True:
            question_enrich_dict = read_question_enrich_data(question_enrich_path)
            to_run = [q for q in all_questions if q not in question_enrich_dict]
            cp.run_sentences(to_run, question_enrich_path)
            if not to_run:
                break


def run_test_process(question_path):
    X, question_enrich_dict = get_data(question_path)
    y = [num for num in range(0, 10) for _ in range(10)]
    all_splits = split_to_train_and_test(X, y, 0.2, n_splits=10)

    for s in all_splits:
        s['train'] = enrich_data(s['train'], question_enrich_dict)

    all_f1_scores = []
    for split in all_splits:
        f1 = train_and_evaluate_transformers(split, MODEL_FULL_NAME)
        all_f1_scores.append(f1)


def train_model_process(question_path, saved_model_path, override=False):
    model_path = f'{saved_model_path}/{MODEL_NAME}.bin'
    if not override and os.path.exists(model_path):
        print('model already exists')
        return
    X, question_enrich_dict = get_data(question_path)
    y = [num for num in range(0, 10) for _ in range(10)]
    X, y = enrich_data((X, y), question_enrich_dict)
    cross_validate_and_save_model(MODEL_NAME, MODEL_FULL_NAME, X, y, saved_model_path, n_splits=5)


def train_student_model(question_path, saved_model_path):
    teacher_model_path = f'{saved_model_path}/{MODEL_NAME}.bin'
    student_model_path = f'{saved_model_path}/{MODEL_NAME}_student'

    X, question_enrich_dict = get_data(question_path)
    y = [num for num in range(0, 10) for _ in range(10)]
    X, y = enrich_data((X, y), question_enrich_dict)

    cross_validate_and_save_student_model(
        X,
        y,
        teacher_model_path,
        student_model_path,
        n_splits=5
    )


def prediction_process(docs_to_predict, saved_model_path, student=False, model=None):
    heb_doc2docs = translate_and_enrich(docs_to_predict)
    final_docs_to_predict, indices_dict = get_docs_and_indices(heb_doc2docs)
    model_path = f'{saved_model_path}/{MODEL_NAME}.bin'
    if student:
        model_path = f'{saved_model_path}/{MODEL_NAME}_student'
    predictions, confidence_scores, all_confidence_scores = predict_with_confidence(
        model_path,
        MODEL_FULL_NAME,
        final_docs_to_predict,
        model
    )
    opt_labels = list(all_confidence_scores[0].keys())
    return calculate_results(indices_dict, all_confidence_scores, opt_labels)


def translate_and_enrich(docs_to_predict):
    ts = TranslateSentences()
    docs_english = [ts.return_eng_to_heb(d) for d in docs_to_predict]
    cp = CreateParaphrase()
    heb_doc2docs = {}
    for d, l in zip(docs_to_predict, docs_english):
        heb_doc2docs[d] = []
        cp.run_sentences(l)
        for s in l:
            heb_doc2docs[d].extend(cp.results[s])
        heb_doc2docs[d] = list(set(heb_doc2docs[d]))
        heb_doc2docs[d] = [do for do in heb_doc2docs[d] if do.strip()]
    return heb_doc2docs
