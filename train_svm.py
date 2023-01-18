import pickle

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
import numpy as np
from sklearn import model_selection, naive_bayes, svm
import logging
import time
from sklearn.metrics import accuracy_score, f1_score
from random import shuffle
import os

from util import read_args, init_experiments
from train import write_results

from dataloader import load_dataset, load_domain_assignment


# preprocess texts with lowercasing stopword removal, lemmatization and tokenization
def prepare_text(text, lowercase=True, stopword_removal=True, lemmatization=True):
    preprocessed_text = np.array(text).copy()

    if lowercase:
        preprocessed_text = [entry.lower() for entry in preprocessed_text]

    # tokenization
    preprocessed_text = [word_tokenize(entry) for entry in preprocessed_text]

    # Step - d : Remove Stop words, Non-Numeric and perform Word Stemming / Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    text = []
    for entry in preprocessed_text:
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):

            if stopword_removal and word.lower() in stopwords.words("english"):
                continue

            if lemmatization:
                word = word_Lemmatized.lemmatize(word, tag_map[tag[0]])

            Final_words.append(word)

        text.append(str(Final_words))

    preprocessed_text = text
    return preprocessed_text


def svm_single_experiment(args, log_dir, log_praefix=""):
    experiment_start_time = time.time()

    domain_assignment = load_domain_assignment(args)

    # train module selector
    if args.num_modules <= 1:
        logging.info(f"skip training of module selector because num_modules={args.num_modules}")
    else:
        logging.info("train module selector")

        task = "domain"
        domains_flat = [item for sublist in domain_assignment for item in sublist]
        data_context = load_dataset(args, task, domains_flat)

        # run test set and compute acc_ms and f1_ms
        y_true_ms, y_pred_ms, train_duration, predict_duration = train_and_predict(data_context.train_dataset.df,
                                                                                   data_context.valid_dataset.df,
                                                                                   data_context.test_dataset.df, task)

        write_results(log_dir, f"{log_praefix}train_duration", train_duration)
        write_results(log_dir, f"{log_praefix}train_num_samples", len(data_context.train_dataset))

        acc = accuracy_score(y_true_ms, y_pred_ms)
        f1 = f1_score(y_true_ms, y_pred_ms, average="micro")

        write_results(log_dir, f"{log_praefix}acc_ms", acc)
        write_results(log_dir, f"{log_praefix}f1_ms", f1)
        write_results(log_dir, f"{log_praefix}predict_duration", predict_duration)
        write_results(log_dir, f"{log_praefix}predict_num_samples", len(data_context.test_dataset))

    # train intent detector
    all_y_true_id = None
    all_y_pred_id = []
    for i_module in range(args.num_modules):
        logging.info(f"train intent detector {i_module + 1}/{args.num_modules}")
        task = "intent"
        data_context = load_dataset(args, task, domain_assignment[i_module])

        x = train_and_predict(data_context.train_dataset.df,
                              data_context.valid_dataset.df,
                              data_context.test_dataset.df, task,
                              additional_test_set=data_context.test_dataset_all)

        y_true, y_pred, train_duration, predict_duration, y_true_additional, y_pred_additional, additional_predict_duration = x

        write_results(log_dir, f"{log_praefix}train_duration", train_duration)
        write_results(log_dir, f"{log_praefix}train_num_samples", len(data_context.train_dataset))

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="micro")
        write_results(log_dir, f"{log_praefix}acc_id_{i_module}", acc)
        write_results(log_dir, f"{log_praefix}f1_id_{i_module}", f1)
        write_results(log_dir, f"{log_praefix}predict_duration", predict_duration)
        write_results(log_dir, f"{log_praefix}predict_num_samples", len(data_context.test_dataset_all))

        if args.num_modules > 1:
            # run the complete test set
            all_y_pred_id.append(y_pred_additional)
            all_y_true_id = y_true_additional

    if args.num_modules == 1:
        write_results(log_dir, f"{log_praefix}acc_id_all", acc)
        write_results(log_dir, f"{log_praefix}f1_id_all", f1)
    else:
        # compute final predictions of intent detection
        assert len(y_true_ms) == len(y_pred_ms)
        assert len(y_true_ms) == len(all_y_true_id)
        for i in range(len(all_y_pred_id)):
            assert len(all_y_pred_id[i]) == len(y_pred_ms)

        domain_id = {}
        for i in range(len(domain_assignment)):
            for domain in domain_assignment[i]:
                domain_id[domain] = i

        y_pred = []
        for i in range(len(y_true_ms)):
            module = domain_id[y_pred_ms[i]]
            y_pred.append(all_y_pred_id[module][i])

        acc = accuracy_score(all_y_true_id, y_pred)
        f1 = f1_score(all_y_true_id, y_pred, average="micro")
        write_results(log_dir, f" {log_praefix}acc_id_all", acc)
        write_results(log_dir, f"{log_praefix}f1_id_all", f1)

    duration = (time.time() - experiment_start_time)
    write_results(log_dir, f" {log_praefix}total_duration", duration)


def train_and_predict(train_dataset, valid_dataset, test_dataset, task, additional_test_set=None):

    # preprocess features
    pickle_dataset = "store"
    assert pickle_dataset in ("none", "store", "load")

    train_x = None
    valid_x = None
    test_x = None
    additional_test_x = None
    pickle_path = f"/tmp/dataset_{task}.pkl"
    if pickle_dataset in ("none", "store"):
        logging.info("preprocessing data")
        train_x = prepare_text(train_dataset.text)
        valid_x = prepare_text(valid_dataset.text)
        test_x = prepare_text(test_dataset.text)
        if additional_test_set is not None:
            additional_test_x = prepare_text(additional_test_set.df.text)

        tfidf_vect = TfidfVectorizer(max_features=5000)
        tfidf_vect.fit(train_x)

        train_x = tfidf_vect.transform(train_x)
        valid_x = tfidf_vect.transform(valid_x)
        test_x = tfidf_vect.transform(test_x)

        if additional_test_x is not None:
            additional_test_x = tfidf_vect.transform(additional_test_x)

        if pickle_dataset == "store":
            pickle.dump((train_x, valid_x, test_x, additional_test_x), open(pickle_path, "wb"))
    if pickle_dataset == "load":
        logging.info("use pickled dataset")
        train_x, valid_x, test_x, additional_test_x = pickle.load(open(pickle_path, "rb"))

    # preprocess labels
    encoder = LabelEncoder()
    encoder = encoder.fit(np.unique(train_dataset[task]))
    train_y = encoder.transform(train_dataset[task])

    if len(set(train_y)) == 1:
        # svm does not work for a single class only.
        # in this case we do not train anything and predict this class always
        y_pred = [train_y[0]] * test_x.shape[0]
        y_pred = encoder.inverse_transform(y_pred)
        y_true = test_dataset[task]
        train_duration = 0
        predict_duration = 0

        if additional_test_x is not None:
            y_true_additional = additional_test_set.df[task]
            y_pred_additional = [train_y[0]] * len(y_true_additional)
            y_pred_additional = encoder.inverse_transform(y_pred_additional)
            additional_predict_duration = 0
            return y_true, y_pred, train_duration, predict_duration, y_true_additional, \
                y_pred_additional, additional_predict_duration
        else:
            return y_true, y_pred, train_duration, predict_duration

    # shuffle training data
    ind_list = list(range(len(train_y)))
    np.random.shuffle(ind_list)
    train_y = train_y[ind_list]
    train_x = train_x[ind_list]

    model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    logging.info("train model")

    start_time = time.time()
    model.fit(train_x, train_y)

    path = "/tmp/svm.pkl"
    pickle.dump(model, open(path, "wb"))
    logging.info(f"model size {os.path.getsize(path)} byte")

    train_duration = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(test_x)
    predict_duration = time.time() - start_time

    y_pred = encoder.inverse_transform(y_pred)
    y_true = test_dataset[task]

    # process additional test data with the model
    y_true_additional = None
    y_pred_additional = None
    if additional_test_x is not None:
        start_time = time.time()
        y_pred_additional = model.predict(additional_test_x)
        additional_predict_duration = time.time() - start_time
        y_pred_additional = encoder.inverse_transform(y_pred_additional)
        y_true_additional = additional_test_set.df[task]
        return y_true, y_pred, train_duration, predict_duration, y_true_additional, \
               y_pred_additional, additional_predict_duration
    else:
        return y_true, y_pred, train_duration, predict_duration


if __name__ == "__main__":

    args = read_args()
    assert args.experiment == "single_experiment"

    name = f"dataset={args.dataset}"
    name += f"..num_modules={args.num_modules}"
    name += f"..add_adapter={args.add_adapter}"
    name += f"..model={args.model_name_or_path}"
    if args.subsample > 0:
        name += "..subsample"
    log_dir = init_experiments(args, name + "..single_experiment")
    svm_single_experiment(args, log_dir)
