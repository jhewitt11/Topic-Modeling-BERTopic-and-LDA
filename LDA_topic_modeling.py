import time
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from tools import read_in_data
from tools import LDA_topic_report
from tools import classification_matrix


directory = 'data/clean/'

documents, labels, category_freqs =  read_in_data(directory+'BBC_data_CLEAN_P.csv', x_col = 1, y_col = 2)

train_docs, test_docs, train_labels, test_labels = train_test_split(documents, 
                                                                    labels,
                                                                    test_size = 0.1,
                                                                    random_state = 0)

pipeline = Pipeline([
            ('CountVectorizer', CountVectorizer()),
            ('TFIDF_transformer', TfidfTransformer()),
            ('LDA', LatentDirichletAllocation())
])

parameter_grid = {
    'CountVectorizer__max_df' : [0.95, 0.9, 0.85,],
    'CountVectorizer__min_df' :  [0.1, 0.15, 0.2],
    'CountVectorizer__ngram_range' : [(1, 2)],

    'LDA__n_components' : [5],
    'LDA__doc_topic_prior' : [0.1, 0.05, 0.01, 0.005 ],
    'LDA__topic_word_prior' : [0.1, 0.2, 0.3],
    'LDA__max_iter' : [100, 250]
}


# original best params in report
best_parameter_grid = {
    'CountVectorizer__max_df': 0.85, 
    'CountVectorizer__min_df': 0.1, 
    'CountVectorizer__ngram_range': (1, 1), 
    
    'LDA__n_components': 5, 
    'LDA__doc_topic_prior': 0.005, 
    'LDA__topic_word_prior': 0.1,
    'LDA__max_iter' : 250,
}


grid_searcher = GridSearchCV(
                    estimator = pipeline,
                    param_grid = parameter_grid,
                    refit = True,
                    verbose = 1,
                    )

'''
grid_searcher.fit(train_docs)
print(f'\nBest score found : {grid_searcher.best_score_}\nBest parameters  : {grid_searcher.best_params_}')
print(f'Best parameters set.')
pipeline = grid_searcher.best_estimator_
'''


# Training / eval without Gridsearch
pipeline.set_params(**best_parameter_grid)
topic_probs = pipeline.fit_transform(train_docs)
top_topics = np.argmax(topic_probs, axis = 1)
log_prior_score = pipeline.score(test_docs)


# index to word dictionary
ind_2_word = {ind : word for ind, word in enumerate(pipeline['CountVectorizer'].get_feature_names_out())}


# print report
LDA_topic_report(pipeline['LDA'], ind_2_word, j = 7)


results, results_dict = classification_matrix(train_labels, 
                                              category_freqs,
                                              top_topics, 
                                              )
                                              
print(f'Log Prior Score : {log_prior_score:.3f}')


























# BERTTopic
## any preprocessing?
## what other decisions? 

## .fit
## transform for new inference?

# Results
## compare top_n_words
## data visualization

## BERT vs. LDA topic #
