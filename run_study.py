import joblib
import pandas as pd
import numpy as np

from tools import read_in_data
from tools import model_topics_in_batch
from tools import create_report

read_directory = 'data/clean/'
result_directory = 'results/'

documents, labels, categories =  read_in_data(read_directory+'BBC_data_CLEAN.csv',   
    x_col = 1,  # column holding documents                              
    y_col = 2   # column holding labels
)

embeddings = joblib.load(read_directory + 'BBC_embeddings.z')

# Model Selection
model_name = 'BERTopic'
#model_name = 'LDA'

# CountVectorizer parameters
CV_grid = {
    'max_df': 0.85, 
    'min_df': 0.1, 
    'ngram_range': (1, 1), 
}

# LDA parameters
LDA_grid = {    
    'LDA__n_components': 15, 
    'LDA__doc_topic_prior': 0.5, 
    'LDA__topic_word_prior': 0.5,
    'LDA__max_iter' : 100,
}

# BERTopic parameters
min_topic_size = 10

model_params = {
    'CV_grid' : CV_grid,
    'LDA_grid' : LDA_grid,
    'min_topic_size' : min_topic_size,

}

# General parameters
iters = 8
N = 10
ID = np.random.randint(low = 0, high = 1000)


result_df = model_topics_in_batch(
    model_name, 	# LDA or BERTopic
    documents, 		# cleaned document files
    embeddings, 	# embeddings for document files
    labels, 		# category for each document
    categories, 	# list of different categories in labels
    iters, 			# number of iterations to run
    N, 				# parameter for topic coherence, top N words from a topic are analyzed 
    **model_params	# parameters for LDA and BERTopic models
)

create_report(
    model_name,         # LDA or BERTopic
    ID,                 # randomly generated ID
    iters,              # number of iterations
    result_directory,   # write location for report
    result_df,          # output from model_topics_in_batch
)
