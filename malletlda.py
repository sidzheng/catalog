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
    df = df.loc[df['year'] > 1999]

    # Convert to list
    # data = df.abstract.values.tolist()
    data = df.abstract.values.tolist() + df.title.values.tolist() + df.keywords.values.tolist()
    data = [re.sub('\[\S*\]?', '', i) for i in data]
    data = [re.sub('\【\S*\】?', '', i) for i in data]
    data = [i for i in data if i != '']
    return df, data


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


def remove_stopwords(texts):
    stop_words = read_stopwords()
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    texts_nostops = remove_stopwords(texts)
    return [bigram_mod[doc] for doc in texts_nostops]


def compute_coherence_values(dictionary, corpus, texts, start=5, stop=45, step=5):
    coherence_values = []
    model_list = []
    max_ldacor = {'num_topics': 0, 'ldacor': 0}
    mallet_path = 'mallet'
    for num_topics in range(start, stop, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherencemodel.get_coherence()
        coherence_values.append(coherence_lda)
        if coherence_lda > max_ldacor['ldacor']:
            max_ldacor['ldacor'] = coherence_lda
            max_ldacor['num_topics'] = num_topics
            max_ldacor['model'] = model

    return model_list, coherence_values, max_ldacor


def show_ldacor(coherence_values, start=5, stop=45, step=5):
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig('malletlda_cor.png')
    plt.show()
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


if __name__ == '__main__':
    data_path = r'.\detailthesis'
    df, data_lst = csv_to_df(data_path)

    # lst = kw_lst(df)

    data_words = jieba_seg(data_lst)

    # Form Bigrams and remove stopwords
    data_words_bigrams = make_bigrams(data_words)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words_bigrams)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_words_bigrams]

    model_list, coherence_values, max_ldacor = compute_coherence_values(id2word, corpus, data_words_bigrams)
    print(max_ldacor)
    show_ldacor(coherence_values)

    # Select the model and print the topics
    optimal_model = max_ldacor['model']
    model_topics = optimal_model.show_topics(formatted=False)
    pprint(optimal_model.print_topics(num_words=10))

    malletmodel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model)
    vis = pyLDAvis.gensim.prepare(malletmodel, corpus, id2word)
    pyLDAvis.save_html(vis, 'malletlda.html')
