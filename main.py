import os
import re
import json
import string
import sys
import warnings
import argparse
from collections import defaultdict
from functools import reduce
import textwrap

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from keras.engine.input_layer import Input
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.layers.merge import Concatenate
from keras.engine.input_layer import Input
from keras.optimizers import SGD, Adam
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import one_hot

STOP_WORDS = ["i", "me", "my", "myself", "we", "our",
              "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she",
              "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what",
              "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were",
              "be", "been", "being", "have", "has", "had",
              "having", "do", "does", "did", "doing", "a",
              "an", "the", "and", "but", "if", "or",
              "because", "as", "until", "while", "of", "at",
              "by", "for", "with", "about", "against", "between",
              "into", "through", "during", "before", "after", "above",
              "below", "to", "from", "up", "down", "in",
              "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when",
              "where", "why", "how", "all", "any", "both",
              "each", "few", "more", "most", "other", "some",
              "such", "no", "nor", "not", "only", "own",
              "same", "so", "than", "too", "very", "s",
              "t", "can", "will", "just", "don", "should",
              "now"]

URL_MARKER = "%url%"
IMG_MARKER = "%img%"


def setup():
    warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


def read_data(data_path, train=True):
    def read_mails(data_path, label_type=None):
        instances = []
        if label_type:
            prefix = os.path.join(data_path, label_type)
        else:
            prefix = data_path
        print(prefix)
        # Get files from folder recursively
        all_files = []
        for root, _, files in os.walk(prefix):
            file_paths = list(map(lambda p: os.path.join(root, p), files))
            all_files.extend(file_paths)
        messages = map(lambda p: (os.path.basename(p), open(p, encoding='ISO-8859-1').read()),
                       all_files)
        for mail_id, msg in messages:
            subject, content = msg.split('\n', 1)
            # Remove subject title
            subject = subject[len("Subject:"):]
            if label_type:
                label = 1 if label_type == 'spam' else 0
                instance = {
                    'subject': subject,
                    'content': content,
                    'label': label,
                    'id': mail_id
                }
            else:
                instance = {
                    'subject': subject,
                    'content': content,
                    'id': mail_id
                }
            instances.append(instance)
        return instances

    if train:
        # train data
        instances = read_mails(data_path, "spam")
        instances.extend(read_mails(data_path, "clean"))
        return instances
    else:
        # test data
        instances = read_mails(data_path)
        return instances


def parse_data(data, parse_html=True):
    """Emails are split into subject and body. I must extract the tokens
    from both inputs and remove elements which offer no potential information
    gain: stopwords, punctuation marks etc."""

    # Define regex patterns for tokenizer
    # markers
    marker_str = URL_MARKER + "|" + IMG_MARKER
    # base64 elements
    base64_str = r'^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?$'
    # emoticons
    emoticons_str = r"(?:[:=;][oO\-\^]?[D\)\]\(\[/\\OpPo])"
    # HTML tags
    html_str = r'<[^>]+>'
    # URLs
    url_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f]))+'
    # numbers
    number_str = r'(?:(?:\d+,?)+(?:\.?\d+)?)'
    # words
    words_str = r"(?:[a-z][a-z'\-_]+[a-z])"
    # other words
    other_w_str = r'(?:[\w_]+)'
    # anything else
    others_str = r'(?:\S)'

    # compile regexes
    regexes = [marker_str, html_str, base64_str, emoticons_str, url_str,
               number_str, words_str, other_w_str, others_str]

    tokens_re = re.compile(r'(' + '|'.join(regexes) + ')',
                           re.VERBOSE | re.IGNORECASE)
    url_re = re.compile(r'^' + url_str + '$', re.VERBOSE | re.IGNORECASE)

    # keep: $?!
    filtered_punctuation = ""  # "<>()[]{}/\|-_#%^&*,.:;@~`+=\""

    def preprocess(text, lowercase=False, parse_html=True):
        if parse_html:
            try:
                soup = BeautifulSoup(text, "html.parser")
                # Replace link HTML tags with markers.
                for tag in soup.find_all('a'):
                    tag.replace_with(URL_MARKER)
                # Replace image HTML tags with markers.
                for tag in soup.find_all('img'):
                    tag.replace_with(IMG_MARKER)
                clean_body = soup.text
            # Sometimes bs throws a type error. In that case, just skip HTML
            # parsing.
            except TypeError:
                print("parse_data.preprocess: Error while processing HTML. Skipping...")
                clean_body = text
        else:
            clean_body = text
        tokens = tokens_re.findall(clean_body)
        # Replace direct links with markers and strip punctuation
        # and stop words.
        processed_tokens = []
        for token in tokens:
            processed = token[0]
            if lowercase:
                processed = token[0].lower()
            if token[0] in filtered_punctuation:
                continue
            elif token[0] in STOP_WORDS:
                continue
            elif url_re.match(token[0]):
                processed = URL_MARKER
            processed_tokens.append(processed)
        return processed_tokens

    proc_instances = []
    for inst in data:
        subject = inst["subject"]
        content = inst["content"]
        proc_subject = preprocess(subject, lowercase=True, parse_html=parse_html)
        proc_content = preprocess(content, lowercase=True, parse_html=parse_html)
        proc_instance = dict()
        proc_instance["subject"] = ' '.join(proc_subject)
        proc_instance["content"] = ' '.join(proc_content)
        if inst.get("label") in [0, 1]:
            proc_instance["label"] = inst["label"]
        proc_instance["id"] = inst["id"]
        proc_instances.append(proc_instance)
    return proc_instances


def extract_features(data, get_ids=False, subject_features=40, content_features=200):
    """ This function should extract the features of subject and body
    *separately*. Possible feature extraction methods: TF-IDF, bag of
    words."""
    def TF_IDF(data, max_features=50, dense=True):
        vectorizer = TfidfVectorizer(norm='l2', use_idf=True,
                                     smooth_idf=True, sublinear_tf=True,
                                     ngram_range=(1,3), max_features=max_features)
        if dense:
            return vectorizer.fit_transform(data).todense()
        return vectorizer.fit_transform(data)

    def bag_words(data, max_features=50, dense=True):
        vectorizer = CountVectorizer(ngram_range=(1,3), analyzer="word",
                                     max_features=max_features, binary=False,
                                     strip_accents=None)
        if dense:
            return vectorizer.fit_transform(data).todense()
        return vectorizer.fit_transform(data)

    subjects = []
    contents = []
    labels = []
    ids = []
    # Check if labels have been attached
    labels_exist = bool(data[0].get("label"))
    for instance in data:
        subjects.append(instance["subject"])
        contents.append(instance["content"])
        if labels_exist:
            labels.append(instance["label"])
        if get_ids:
            ids.append(instance["id"])
    subjects_matrix = TF_IDF(subjects, subject_features)
    contents_matrix = TF_IDF(contents, content_features)
    features = np.hstack((contents_matrix, subjects_matrix))
    ret_values = []
    ret_values.append(features)
    if labels_exist:
        ret_values.append(np.array(labels))
    if get_ids:
        ret_values.append(ids)
    return tuple(ret_values)


def embedding_extraction(data, vocab_size=1000, get_ids=False):
    messages = []
    labels = []
    ids = []
    # Check if labels have been attached
    labels_exist = bool(data[0].get("label"))
    for instance in data:
        messages.append(' '.join([instance["subject"], instance["content"]]))
        if labels_exist:
            labels.append(instance["label"])
        if get_ids:
            ids.append(instance["id"])
    encoded = [one_hot(msg, vocab_size) for msg in messages]
    ret_values = []
    ret_values.append(encoded)
    if labels_exist:
        ret_values.append(np.array(labels))
    if get_ids:
        ret_values.append(ids)
    #print(max([len(msg.split(' ')) for msg in messages]))
    #print(len(set(word for word in ' '.join(messages).split(' '))))
    #print(json.dumps(encoded_docs, indent=4))
    return tuple(ret_values)


def cnn_lstm(input_dim):
    """
    Create a CNN model with word embeddings.
    """
    model_input = Input(shape=input_dim)
    #net = Embedding(input_dim=
    net = Dropout(0.5)(model_input)
    conv_blocks = []
    for sz in [3, 8]:
        conv = Conv1D(filters=10,
                      kernel_size=sz,
                      padding='valid',
                      activation='relu',
                      strides=1)(net)
        conv = AveragePooling1D(pool_size=2)(conv)
        conv = Bidirectional(LSTM(64))(conv)
        #conv = Flatten()(conv)
        conv_blocks.append(conv)
    net = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    net = Dropout(0.8)(net)
    net = Dense(15, activation='relu')(net)
    model_output = Dense(1, activation='sigmoid')(net)
    model = Model(model_input, model_output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0004), metrics=['acc'])
    model.summary()
    return model


def build_classifier(classifier, data_shape=None):
    """I may need to test different classifiers, so I should make it as
    generic as possible."""
    clf_formula = classifier.lower()
    clf_list = []
    for clf in re.findall(r"\w+", clf_formula):
        if clf == "svmrbf":
            clf_list.append((clf, SVC(C=1.0, gamma="scale",
                                      kernel="rbf", probability=True)))
        elif clf == "svmlin":
            clf_list.append((clf, SVC(C=1.0, gamma="scale",
                                      kernel="linear", probability=True)))
        elif clf == "logreg":
            clf_list.append((clf, LogisticRegression()))
        elif clf == "multinb":
            clf_list.append((clf, MultinomialNB(alpha=3.0)))
        elif clf == "rfc":
            clf_list.append((clf, RandomForestClassifier(192, "entropy")))
        elif clf == "cnnlstm":
            if data_shape is None:
                raise ValueError("Must specify data_shape for CNN-LSTM")
            else:
                clf_list.append(cnn_lstm(data_shape))
        else:
            raise ValueError("Could not find {}".format(clf))
    if len(clf_list) == 0:
        raise ValueError("Classifier string invalid: {}".format(clf_formula))
    elif len(clf_list) == 1:
        return clf_list[0][1]
    else:
        ensemble = VotingClassifier(clf_list, voting='soft')
        return ensemble


#def build_classifier(classifier):
#    """I may need to test different classifiers, so I should make it as
#    generic as possible."""
#    clf = classifier.lower()
#    if clf == "svmrbf":
#        return SVC(C=1.0, gamma="scale", kernel="rbf")
#    elif clf == "svmlin":
#        return SVC(C=1.0, gamma="scale", kernel="linear")
#    elif clf == "logreg":
#        return LogisticRegression()
#    elif clf == "multinb":
#        return MultinomialNB(alpha=3.0)
#    elif clf == "rfc":
#        return RandomForestClassifier(192, "entropy")
#    # TODO: Add more models: SVM, Random forest, MLP network
#    else:
#        raise ValueError("Could not find {}".format(classifier))


def split_data(input_data, labels, train_split=0.8, shuffle=True):
    indices = np.arange(0, labels.size)
    if shuffle:
        indices = np.random.permutation(indices)
    split_point = int(train_split * labels.size)
    train_x = input_data[indices[:split_point], :]
    test_x = input_data[indices[split_point:], :]
    train_y = labels[indices[:split_point]]
    test_y = labels[indices[split_point:]]
    return train_x, train_y, test_x, test_y


def train_classifier(classifier, features, labels):
    """This function must concern itself with training the classifier
    on the specified data."""
    return classifier.fit(features, labels)


def simple_classifier_score(clf, features, labels, shuffle=True):
    """A simple function that tests classifiers and computes the mean
    test accuracy."""
    print("Spliting data...")
    train_x, train_y, test_x, test_y = split_data(features, labels, shuffle=shuffle)
    print("Training clasifier...")
    trained_clf = train_classifier(clf, train_x, train_y)
    print("Testing classifier...")
    return trained_clf.score(test_x, test_y)


def cross_validation(classifier, features, labels, folds=5, metrics="acc"):
    """Do a cross-validation on the specified data and return results
    for the desired metrics."""
    all_scores = defaultdict(list)
    preds = []
    kf = KFold(n_splits=folds, shuffle=True)
    for train_index, test_index in kf.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        train_classifier(classifier, x_train, y_train)
        y_pred = classifier.predict(x_test)
        if metrics is not None:
            if isinstance(metrics, str):
                ms = [metrics]
            elif isinstance(metrics, list):
                ms = metrics
            fold_scores = compute_metrics(y_test, y_pred, metrics=ms)
            for k in fold_scores.keys():
                all_scores[k].append(fold_scores[k])
    return all_scores if all_scores.keys() else None


def compute_metrics(test_y, pred_y, metrics=["acc"]):
    scores = {}
    for metric in metrics:
        if metric == "acc":
            scores["acc"] = accuracy_score(test_y, pred_y)
        elif metric == "conf_mat":
            scores["conf_mat"] = confusion_matrix(test_y, pred_y)
        elif metric == "prec":
            scores["prec"] = precision_score(test_y, pred_y)
        elif metric == "rec":
            scores["rec"] = recall_score(test_y, pred_y)
        elif metric == "f1":
            scores["f1"] = f1_score(test_y, pred_y)
    return scores


def extract_metrics(metrics):
    acc = np.array(metrics["acc"]).mean()
    f1 = np.array(metrics["f1"]).mean()
    recall = np.array(metrics["rec"]).mean()
    precision = np.array(metrics["prec"]).mean()
    conf_mats = metrics["conf_mat"]
    conf_total = reduce(lambda X1, X2: X1 + X2, conf_mats)
    tn = conf_total[0,0]
    fp = conf_total[0,1]
    fn = conf_total[1,0]
    tp = conf_total[1,1]
    return {
        "acc": acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "rec": recall,
        "prec": precision,
        "f1": f1
    }


def create_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', action="store_true",
                        help="show this message and quit")
    subparsers = parser.add_subparsers(dest="command")
    parser_info = subparsers.add_parser("info",
                                        help="display useful information")
    parser_info.add_argument("outfile", action="store",
                             help="path to output file")
    parser_scan = subparsers.add_parser("scan",
                                        help="classify emails in a given directory")
    parser_scan.add_argument("directory", action="store",
                             help="path to email directory")
    parser_scan.add_argument("outfile", action="store",
                             help="path to output file")
    parser_scan.add_argument("--model-name", action="store",
                             help="name of model to load",
                             default="rfc170")
    parser_scan.add_argument("--with-labels", action="store_true",
                             help="parse directory contents as \
                                     during the training phase")
    parser_eval = subparsers.add_parser("eval",
                                        help="compute metrics for classifier")
    parser_eval.add_argument("directory", action="store",
                             help="path to email directory")
    parser_eval.add_argument("--model-name", action="store",
                             help="name of saved model",
                             default="model")
    parser_eval.add_argument("--classifier", action="store",
                             help="name of classifier",
                             default="rfc")
    group_processed = parser_eval.add_mutually_exclusive_group()
    group_processed.add_argument("--save-processed", action="store",
                                 dest="processed_savepath",
                                 help="save processed data")
    group_processed.add_argument("--load-processed", action="store",
                                 dest="processed_loadpath",
                                 help="load processed data")
    group_features = parser_eval.add_mutually_exclusive_group()
    group_features.add_argument("--save-features", action="store",
                                dest="features_savepath",
                                help="save extracted features and labels")
    group_features.add_argument("--load-features", action="store",
                                dest="features_loadpath",
                                help="load extracted features and labels")
    return parser


def main():
    setup()
    parser = create_parser()
    # Modify command-line arguments in order to make
    # them compatible with argparse
    sys.argv = [re.sub(r'^-(info|scan|eval)', r'\g<1>', arg)
                for arg in sys.argv]
    args = parser.parse_args()
    # Change help message in order to print commands
    # preceeded by a dash
    if args.help:
        help_string = parser.format_help()
        help_string = re.sub(r'(info|scan|eval)', r'-\g<1>', help_string)
        print(help_string)
    elif args.command is not None:
        command = args.command
        if command == "info":
            info_msg = textwrap.dedent("""\
                SpamAway
                Ghiga Claudiu-Alexandru
                gca22
                0.2
            """)
            open(args.outfile, "w").write(info_msg)
        elif command == "scan":
            print("Scanning directory {}".format(args.directory))
            raw_data = read_data(args.directory, train=args.with_labels)
            print("Processing data...")
            processed_data = parse_data(raw_data, parse_html=False)
            print("Extracting features...")
            values = extract_features(processed_data,
                                      get_ids=True,
                                      subject_features=10,
                                      content_features=80)
            input_data = values[0]
            ids = values[len(values) - 1]
            print("Loading model...")
            trained_model = joblib.load(os.path.join("models", args.model_name + ".sav"))
            preds = trained_model.predict(input_data)
            print("Writing to output file...")
            f = open(args.outfile, "w")
            for i in range(len(preds)):
                label = "cln" if preds[i] == 0 else "inf"
                line = "{}|{}\n".format(ids[i], label)
                f.write(line)
            f.close()
            if args.with_labels:
                labels = values[1]
                metrics = ["acc", "conf_mat", "rec", "prec", "f1"]
                print(compute_metrics(labels, preds, metrics=metrics))
            print("Done.")
        elif command == "eval":
            print("Training and evaluating model...")
            print("-"*40)
            print("Reading data...")
            raw_data = read_data(args.directory, train=True)
            if args.processed_loadpath is not None:
                path = args.processed_loadpath
                filepath = os.path.join(path, 'processed.json')
                print("Loading processed data located at {}".format(path))
                try:
                    processed_data = json.load(open(filepath))
                except FileNotFoundError:
                    print("main.load_processed: no such file exists")
                    parser.exit(1)
            else:
                print("Processing data...")
                processed_data = parse_data(raw_data, parse_html=False)
            if args.processed_savepath is not None:
                path = args.processed_savepath
                print("Saving processed data at {}".format(path))
                if not os.path.exists(path) or not os.path.isdir(path):
                    os.makedirs(path)
                json.dump(processed_data,
                          open(os.path.join(path, 'processed.json'), 'w'),
                          indent=4)
            if args.features_loadpath is not None:
                path = args.features_loadpath
                features_path = os.path.join(path, 'features.npy')
                labels_path = os.path.join(path, 'labels.npy')
                print("Loading features and labels located at {}".format(path))
                try:
                    input_data = np.load(features_path)
                    labels = np.load(labels_path)
                except IOError:
                    print("main.load_features: no such file exists")
                    parser.exit(1)
            else:
                print("Extracting features...")
                #input_data, labels = embedding_extraction(processed_data,
                #                                          vocab_size=412000,
                #                                          get_ids=False)
                input_data, labels = extract_features(processed_data,
                                                      subject_features=10,
                                                      content_features=80)
            if args.features_savepath is not None:
                path = args.features_savepath
                print("Saving features at {}".format(path))
                if not os.path.exists(path) or not os.path.isdir(path):
                    os.makedirs(path)
                np.save(os.path.join(path, 'features'), input_data)
                np.save(os.path.join(path, 'labels'), labels)
            x_train, y_train, x_test, y_test = split_data(input_data, labels, train_split=0.5)
            print("Building classifier...")
            clf = build_classifier(args.classifier)
            print("Cross validation...")
            metrics = ["acc", "conf_mat", "rec", "prec", "f1"]
            scores = cross_validation(clf, x_train, y_train, metrics=metrics)
            print(scores)
            scores = extract_metrics(scores)
            total = y_train.size
            print("-"*40)
            print("RESULTS")
            results_msg = textwrap.dedent("""\
            Total: {total} predictions of which:
                True negatives: {tp}
                False positives: {fp}
                False negatives: {fn}
                True positives: {tn}
            Accuracy: {acc}
            Recall: {rec}
            Precision: {prec}
            F1 score: {f1}
            """)
            print(results_msg.format(total=total, **scores))
            print("Training and saving model...")
            clf.fit(x_train, y_train)
            joblib.dump(clf, os.path.join("models", args.model_name + ".sav"))
            print("Testing model...")
            y_pred = clf.predict(x_test)
            scores = compute_metrics(y_test, y_pred, metrics=metrics)
            print(scores)
            print("Done.")


if __name__ == "__main__":
    main()
