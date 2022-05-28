# Author: Stefan Petrescu @ 2022

import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import spacy
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
import gensim.corpora as corpora
from pprint import pprint
from gensim.models import CoherenceModel
import numpy as np
import tqdm
import matplotlib.pyplot as plt


def read_data(path):
    df = pd.read_csv(path)
    return df


def tokenize_words(documents):
    for document in documents:
        yield gensim.utils.simple_preprocess(str(document), deacc=True)  # deacc=True removes punctuations


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def compute_coherence_values(corpus, dictionary, k, a, b, texts):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()


if __name__ == "__main__":
    path = "../data/raw/obs_window.csv"
    print(f"Reading raw data file: {path}")

    documents = read_data(path=path)
    print(documents.head())

    data = documents.Document.values.tolist()
    data_words = list(tokenize_words(data))

    print(data_words[:1][0][:30])

    data_words_nostops = remove_stopwords(data_words)
    print(f"Number of documents analyzed: {len(data_words_nostops)}")

    id2word = corpora.Dictionary(data_words_nostops)

    texts = data_words_nostops

    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)

    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_nostops, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nBaseline coherence score: ', coherence_lda)

    grid = {}
    grid['Validation_Set'] = {}
    min_topics = 8
    max_topics = 10
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    print(topics_range)

    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

    num_of_docs = len(corpus)
    corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.80)), corpus]
    corpus_title = ['80% Corpus', '100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }

    if 1 == 1:
        pbar = tqdm.tqdm(total=540)

        for i in range(len(corpus_sets)):
            for k in topics_range:
                for a in alpha:
                    for b in beta:
                        cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
                                                      k=k, a=a, b=b, texts=data_words_nostops)
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)

                        pbar.update(1)
        pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
        results = pd.DataFrame(model_results)
        print(pd.DataFrame(model_results))
        results = results.sort_values('Coherence').drop_duplicates('Topics', keep='last')
        print(results)

        pbar.close()
        results.plot(x="Topics", y="Coherence")
        plt.show()


