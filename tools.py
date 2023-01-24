import csv
import numpy as np
import string
import re
import pandas as pd
import time
from itertools import combinations

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns
import matplotlib.pyplot as plt

import bertopic
from bertopic import BERTopic

##
## Preprocessing tools
##

# Helper function for clean_text
# https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Performs a number of steps to clean text
def clean_text(document):
    
    #remove special characters
    document = document.encode('ascii', 'ignore').decode()
    
    #replace $ and following numbers with placeholders
    document = re.sub('\$\d+\.*\d+', 'MONEYS', document)
    
    #get rid of new lines
    document = re.sub('\\n', ' ', document)
    
    #remove extra spaces
            
    return document
    
def preprocess_text(document, custom_stopwords):

    #make all characters lowercase
    document = document.lower()

    #remove punctuation
    document = re.sub('[%s]' % re.escape(string.punctuation), '', document)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english') + custom_stopwords
    
    doc_temp = []
    for word in document.split(' '):
        if word not in stop_words :
            doc_temp.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
    
    document = ' '.join(doc_temp)

    return document
    
# Undersamples documents to balance categories
def undersample_dataframe(base_df, column, random_val):
    
    new_df = pd.DataFrame(columns = base_df.columns)
    num_samples = sorted(base_df[column].value_counts())[0]
    
    for unique_class in base_df[column].unique():
        
        class_df = base_df[base_df[column] == unique_class]
        class_df = class_df.sample(n = num_samples, random_state = random_val)
        new_df = pd.concat([class_df, new_df], axis = 0)
        
    return new_df

# Read clean text file for use
def read_in_data(file_name, x_col, y_col):
    with open(file_name, encoding = 'utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')

        categories = {}
        category_freqs = {}
        docs = []
        cat_ind = []
        labels = []

        for i, row in enumerate(reader) :
            if i == 0 : continue

            category = row[y_col]
            doc =  row[x_col]

            docs.append(doc)
            labels.append(category)

            if categories.get(category) == None :
                categories[category] = len(categories)
                category_freqs[category] = 0

            cat_ind.append(categories[category])
            category_freqs[category] += 1

            
    #y_oh = one_hot(cat_ind, len(categories))
        
    return docs, labels, list(category_freqs.keys())


##
## 
##


def BERTopic_topic_report(freq_df, all_topics, j = 10):
    
    for topic_num, count in zip( freq_df['Topic'], freq_df['Count']):

        print(f'topic: {topic_num} document count : { count}')
        probs = all_topics[topic_num]
       
        for k, prob in enumerate(probs):
            if k >= j:
                break
            print('\t', prob)
    
    return

# Print report of top j words in each topic
def LDA_topic_report(lda, ind_2_word, j = 10):
    
    top_sorted_vocabs = np.fliplr(np.argsort(lda.components_, axis = 1)[:, -j:])

    print('\nTopic Report : \n')
    for k, vocab_indexes in enumerate(top_sorted_vocabs) :
        print(f'\ttopic #{k} : ', end = '')
        
        for i, ind in enumerate(vocab_indexes):
            print(f'{ind_2_word[ind]} ({lda.components_[k][ind]:.4f}) | ', end = '')
            
        print('\n')
    
    return

# build result matrix and result dictionary
def classification_matrix(labels, categories, top_topics):

    topics = list(set(top_topics))        
         
    # Build results dictionary
    results_dict = {}    
    for label, topic in zip(labels, top_topics): 
        if results_dict.get((label, topic)):
            results_dict[(label, topic)] += 1
        else:
            results_dict[(label, topic)] = 1
        
    # Fill in results matrix
    results = np.zeros((len(topics), len(categories)))
    for i, category in enumerate(categories):
        for k in topics:
            if results_dict.get((category, k)):
                results[k][i] = results_dict[(category, k)]

        print('\n', end='')
        
    return results, results_dict


# Coherence score for trained BERTopic model.
# NPMI score for each topic.
def BERTopic_coherence_score(topic_model, documents):

    N = 3
    TC_NPMI_scores = []
    str_topic_titles = []

    BT_docs_df = topic_model.get_document_info(documents)
    topic_titles = BT_docs_df['Topic'].unique().tolist()
    topic_titles.sort()
    
    topics_len = len(topic_titles)
    topic_titles.pop(0)
    topic_titles += [-1]
    
    for topic in topic_titles :
      rslt_df = BT_docs_df[BT_docs_df['Topic'] == topic]
      docs = rslt_df['Document'].tolist()
            
      score = calc_topic_coherence(docs, N)
      
      TC_NPMI_scores.append(score)
      str_topic_titles.append(str(topic))
      

    coherence_score_df = {'Topic' : str_topic_titles,
                            'Coherence Score' : TC_NPMI_scores
                            }
                            
    return coherence_score_df
    

# build topic model report PDF
def model_topics_in_batch(documents, 
                            labels,
                            embeddings, 
                            categories, 
                            iters = 1, 
                            set_topic_num = False, 
                            result_dir = False):
                            

    fig, axes = plt.subplots(iters, 
                            2, 
                            squeeze = False, 
                            gridspec_kw = {'hspace' : 0.5}, 
                            **{'figsize' : (12, 9)})


    for i in range(iters):

        # TODO : define parameters outside of function and pass in
        topic_model = BERTopic(min_topic_size = 30,)
        
        topics, probs = topic_model.fit_transform(documents, embeddings)
        
        results, results_dict = classification_matrix(
                                                    labels, 
                                                    categories,
                                                    topics, 
                                                    )

        
        # topic classification visual
        result_df = pd.DataFrame(results, columns = categories ).transpose()
        result_df.set_axis([*result_df.columns[:-1], '-1'], axis = 1, inplace = False)
        
        sns.heatmap(   
                result_df, 
                annot = True,
                fmt = '.0f',
                robust = True,
                square = False,
                yticklabels = True,
                cbar = False,
                ax = axes[i][0],)
        
        axes[i][0].set_ylabel('#'+str(i))
        box = axes[i][0].get_position()
        box.x0 = box.x0 + 0.045
        box.x1 = box.x1 + 0.045
        axes[i][0].set_position(box)
        
        # coherence visual
        
        coherence_score_df = BERTopic_coherence_score(topic_model, documents)
        
        
        sns.barplot( 
                data = coherence_score_df, 
                x = 'Topic', 
                y = 'Coherence Score',
                ax = axes[i][1]
                )
        
        box = axes[i][1].get_position()
        box.x0 = box.x0 + 0.065
        box.x1 = box.x1 + 0.065
        axes[i][1].set_position(box)
        
        

    fig.suptitle('BERTopic assignments and NPMI Coherence scores. BBC data, ' + str(iters) + ' runs')
    fig.text(0.5, 0.04, 'BERTopic assignments', ha = 'center')
    fig.text(0.03, 0.5, 'BERTopic runs', va = 'center', rotation = 'vertical')
    fig.savefig(result_dir + 'results.png')
    
    return
    



##
## Topic Coherence tools
##

# returns shape (1, N)
def top_N(X, N):

    # just 1's and 0's so no double counting
    X_occurences = np.logical_or(X, np.zeros((X.shape))) * 1

    # sum
    word_counts = X_occurences.sum(axis = 0)

    top_word_ind = np.flip(np.argsort(word_counts))[:N]

    return top_word_ind
    
# returns shape (1, vocab_length)
def get_word_probs(X):

    # just 1's and 0's 
    X_occurences = np.logical_or(X, np.zeros((X.shape))) * 1

    # sampling documents within a topic, P(it contains this word)
    w_prob = (X_occurences.sum(axis = 0)) / X.shape[0]

    return w_prob
    
def cooccurence_prob_ji(X, j, i):

    # just 1's and 0's 
    X_occurences = np.logical_or(X, np.zeros((X.shape))) * 1

    i_occur = X_occurences[:,i]
    j_occur = X_occurences[:,j]

    return (np.logical_and(i_occur, j_occur) * 1).sum(axis = 0) / X.shape[0]
    
def calc_topic_coherence(docs, N):

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    X = X.toarray()

    words = vectorizer.get_feature_names_out()
    X_occurences = np.logical_or(X, np.zeros((X.shape))) * 1

    top_n_words_ind = top_N(X, N)
    w_probs = get_word_probs(X)

    sum = 0
    for i, j in combinations(top_n_words_ind, 2):

        pw_j_i = cooccurence_prob_ji(X, j, i)

        if pw_j_i == 0:
            npmi = -1
        else :
            npmi =  np.log( pw_j_i / w_probs[j] / w_probs[i])   /   np.log(pw_j_i)   *   -1

        sum += npmi

    return sum