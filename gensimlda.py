# coding = utf-8

import os
import re
import numpy as np
import pandas as pd
from pprint import pprint
import jieba

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt


# # Enable logging for gensim - optional
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
#
# import warnings
# warnings.filterwarnings("ignore",category=DeprecationWarning)

def select_year(s):
    result = re.search('[0-9]{4}', s).group()
    return int(result)


def csv_to_df(path):
    df = pd.DataFrame(columns=('title', 'keywords', 'abstract'))
    for info in os.listdir(path):
        dom = os.path.abspath(path)  # 获取文件夹的路径
        info = os.path.join(dom, info)  # 将路径与文件名结合起来就是每个文件的完整路径
        data = pd.DataFrame(columns=('title', 'keywords', 'abstract'))
        data = pd.read_csv(info, header=None, names=('title', 'keywords', 'abstract', 'year'), usecols=[0, 3, 4, 5],
                           encoding='utf-8',
                           low_memory=False)
        df = df.append(data)
    df['title'] = df['title'].apply(str)
    df['keywords'] = df['keywords'].apply(str)
    df['abstract'] = df['abstract'].apply(str)
    df['year'] = df['year'].apply(select_year)
    df = df.loc[df['year'] > 2009]
    return df


def kw_lst(df: pd.DataFrame):
    lst = []
    for i in df['keywords']:
        kw = i.split(';')
        for j in kw:
            if j:
                lst.append(j)
    lst = list(set(lst))
    return lst


def jieba_seg(sen_lst: list):
    jieba.load_userdict('kw.txt')
    lst = []
    for i in sen_lst:
        lst.append(list(jieba.cut(i, cut_all=False)))
    return lst


def read_stopwords():
    s = open('stopwords.txt', 'r', encoding='utf-8')
    return s.read().split("\n")


def main():
    data_path = r'.\thesis'
    df0 = csv_to_df(data_path)
    # print(df0)
    # lst = kw_lst(df0)
    # print(lst)
    stop_words = read_stopwords()
    # Convert to list
    data = df0.abstract.values.tolist()
    data = [re.sub('\[\S*\]?', '', i) for i in data]
    data = [re.sub('\【\S*\】?', '', i) for i in data]
    data = [i for i in data if i != '']

    data_words = jieba_seg(data)

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words_bigrams)

    # Create Corpus
    texts = data_words_bigrams

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    lst_ldacor = []
    max_ldacor = {'num_topics': 0, 'ldacor': 0}

    for n in list(range(5, 45, 5)):
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=n,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

        # doc_lda = lda_model[corpus]

        # Compute Perplexity
        print('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        lst_ldacor.append(coherence_lda)
        print('Coherence Score: ', coherence_lda)
        if coherence_lda > max_ldacor['ldacor']:
            max_ldacor['ldacor'] = coherence_lda
            max_ldacor['num_topics'] = n
        print(max_ldacor, '/n')

    print(lst_ldacor)
    plt.plot(list(range(5, 45, 5)), lst_ldacor)
    plt.savefig("gensimlda_cor.png")
    plt.show()
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=max_ldacor['num_topics'],
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

    pyLDAvis.save_html(vis, 'gensimlda.html')


if __name__ == '__main__':
    main()
