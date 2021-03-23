# coding = utf-8

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn

df0 = pd.read_csv('./thesis/info-BOOK.csv', header=None, encoding='utf-8')
df = pd.DataFrame()
df['abstract'] = df0[4].apply(str)


# df['abstract'].append(df0[4].apply(str))


def word_cut(text):
    return " ".join(jieba.cut(text))


df["abstract_cutted"] = df.abstract.apply(word_cut)
print(df['abstract_cutted'])

n_features = 1000

tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df=0.5,
                                min_df=10)
tf = tf_vectorizer.fit_transform(df.abstract_cutted)

n_components = 10
lda = LatentDirichletAllocation(n_components=n_components, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(tf)

LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                          evaluate_every=-1, learning_decay=0.7,
                          learning_method='online', learning_offset=50.0,
                          max_doc_update_iter=100, max_iter=50, mean_change_tol=0.001,
                          n_jobs=1, n_components=n_components, perp_tol=0.1, random_state=0,
                          topic_word_prior=None, total_samples=1000000.0, verbose=0)

n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

p = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)

pyLDAvis.save_html(p, 'lda.html')
