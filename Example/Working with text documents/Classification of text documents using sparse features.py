from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def load_dataset(verbose=False, remove=()):
    data_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    data_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    target_names = data_train.target_names

    y_train, y_test = data_train.target, data_test.target

    t0 = time()
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=.5, min_df=5, stop_words='english'
    )
    X_train = vectorizer.fit_transform(data_train.data)
    duration_train = time() - t0

    t0 = time()
    X_test = vectorizer.transform(data_test.data)
    duration_test = time() - t0

    feature_names = vectorizer.get_feature_names_out()

    if verbose:
        data_train_size_mb = size_mb(data_train.data)
        data_test_size_mb = size_mb(data_test.data)

        print('{} documents - {:.2f}MB (training set)'.format(len(data_train.data), data_train_size_mb))
        print('{} documents - {:.2f}MB (test set)'.format(len(data_test.data), data_test_size_mb))
        print('{} categories'.format(len(target_names)))
        print('vectorize training done in {:.3f}s at {:.3f}MB/s'
              .format(duration_train, data_train_size_mb / duration_train))
        print('n_samples: {}, n_features: {}'.format(X_train.shape[0], X_train.shape[1]))
        print('vectorize testing done in {:.3f}s at {:.3f}MB/s'
              .format(duration_test, data_test_size_mb / duration_test))
        print('n_samples: {}, n_features: {}'.format(X_test.shape[0], X_test.shape[1]))

    return X_train, X_test, y_train, y_test, feature_names, target_names


X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(verbose=True)

clf = RidgeClassifier(tol=1e-2, solver='sparse_cg')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)
ax.set_title('Confusion Matrix for {}\n on thr original documents'.format(clf.__class__.__name__))
plt.show()
