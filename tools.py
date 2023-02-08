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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt

import bertopic
from bertopic import BERTopic

##
## Preprocessing tools
##

def get_wordnet_pos(word):
    """
    Helper function for lemmatization
    Map POS tag to first character lemmatize() accepts.
    https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    """
    
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
    #document = re.sub('\$\d+\.*\d+', 'MONEYS', document)
    
    #get rid of new lines
    document = re.sub('\\n', ' ', document)
    
    #remove extra spaces
            
    return document
    
def preprocess_text(document, custom_stopwords):
    '''A series of simple preprocessing steps, then lemmatization.'''

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
    
def undersample_dataframe(base_df, column, random_val):
    '''Undersample documents in order to balance categories.'''
    
    new_df = pd.DataFrame(columns = base_df.columns)
    num_samples = sorted(base_df[column].value_counts())[0]
    
    for unique_class in base_df[column].unique():
        
        class_df = base_df[base_df[column] == unique_class]
        class_df = class_df.sample(n = num_samples, random_state = random_val)
        new_df = pd.concat([class_df, new_df], axis = 0)
        
    return new_df


def read_in_data(file_name, x_col, y_col):
    '''Read in clean text file.'''
    
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
## Topic modeling tools
##

def BERTopic_topic_report(freq_df, all_topics, j = 10):
    '''Print report of top j words in each topic.'''
    
    for topic_num, count in zip( freq_df['Topic'], freq_df['Count']):

        print(f'topic: {topic_num} document count : { count}')
        probs = all_topics[topic_num]
       
        for k, prob in enumerate(probs):
            if k >= j:
                break
            print('\t', prob)
    
    return


def LDA_topic_report(lda, ind_2_word, j = 10):
    '''Print report of top j words in each topic.'''
    
    top_sorted_vocabs = np.fliplr(np.argsort(lda.components_, axis = 1)[:, -j:])

    print('\nTopic Report : \n')
    for k, vocab_indexes in enumerate(top_sorted_vocabs) :
        print(f'\ttopic #{k} : ', end = '')
        
        for i, ind in enumerate(vocab_indexes):
            print(f'{ind_2_word[ind]} ({lda.components_[k][ind]:.4f}) | ', end = '')
            
        print('\n')
    
    return


def classification_matrix(labels, categories, top_topics):
    '''Build results matrix of shape (# of topics, # of categories) as well as results dictionary.'''
    
    topics = list(set(top_topics))
    num_topics = np.max(topics)+1

     
    # Build results dictionary
    results_dict = {}    
    for label, topic in zip(labels, top_topics): 
        if results_dict.get((label, topic)):
            results_dict[(label, topic)] += 1
        else:
            results_dict[(label, topic)] = 1
        
    # Fill in results matrix
    results = np.zeros((num_topics , len(categories)))
    for i, category in enumerate(categories):
        for k in topics:
            if results_dict.get((category, k)):
                results[k][i] = results_dict[(category, k)]
        
    return results, results_dict
    
##
## Topic Coherence tools
##

def top_N(X, N):

    '''
    Returns top N words from a document set.
    
    Words are ranked by the count of documents they appear in as opposed
    to a total count.
    '''

    # just 1's and 0's so no double counting
    X_occurences = np.logical_or(X, np.zeros((X.shape))) * 1

    # sum
    word_counts = X_occurences.sum(axis = 0)

    top_word_ind = np.flip(np.argsort(word_counts))[:N]

    return top_word_ind

def get_word_probs(X):

    '''
    Returns array of word probabilities for topic coherence calculation.
    
    2 options :
    
    'o' option - calculate probability a word occurs in an individual document.
    
    'tf' option - calculate traditional word probability. # of times a word is used / total word count
    '''

    occurrence_or_truefreq = 'o'

    if occurrence_or_truefreq == 'o':
        # just 1's and 0's 
        X_occurences = np.logical_or(X, np.zeros((X.shape))) * 1

        # sampling documents within a topic, P(it contains this word)
        w_prob = (X_occurences.sum(axis = 0)) / X.shape[0]
    
    elif occurrence_or_truefreq == 'tf':
        w_prob = np.sum(X, axis = 0) / np.sum(X)
    
    else:
        assert(occurrence_or_truefreq in ['tf', 'o'])

    return w_prob
    
def cooccurence_prob_ji(X, j, i):

    # just 1's and 0's 
    X_occurences = np.logical_or(X, np.zeros((X.shape))) * 1

    i_occur = X_occurences[:,i]
    j_occur = X_occurences[:,j]

    cooc = (np.logical_and(i_occur, j_occur) * 1).sum(axis = 0) / X.shape[0]

    return cooc
    
def calc_topic_NPMI(docs, N):

    '''
    Calculates topic coherence metric NPMI
    
    Normalized (Pointwise) Mutual Information in Collocation Extraction
    - Gerlof Bouma
    
    https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf
    https://stats.stackexchange.com/questions/140935/how-does-the-logpx-y-normalize-the-point-wise-mutual-information
    '''

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    X = X.toarray()
    
    words = vectorizer.get_feature_names_out()
    X_occurences = np.logical_or(X, np.zeros((X.shape))) * 1

    top_n_words_ind = top_N(X, N)
    w_probs = get_word_probs(X)

    calc_sum = 0
    count = 0
    for i, j in combinations(top_n_words_ind, 2):
    
        pw_j_i = cooccurence_prob_ji(X, j, i)
        pwj = w_probs[j]
        pwi = w_probs[i]
        
        if (pw_j_i == 1):
            npmi = 1

        elif (pw_j_i == 0) and (pwj > 0) and (pwi > 0):
            npmi = -1
            
        else :
            npmi =  np.log( pw_j_i / pwj / pwi)  /   np.log(pw_j_i)   *   -1

        #print(f'npmi : {npmi}')
        calc_sum += npmi
        count += 1

    return calc_sum / count


def model_coherence_score(topics, documents, N):

    '''
    Calculate NPMI score for each topic.
    '''

    TC_NPMI_scores = []
    str_topic_titles = [] 

    topic_doc_d = {
        'Topic' : topics,
        'Document' : documents
    }
    
    topic_doc_df = pd.DataFrame(topic_doc_d)    
    topic_titles = range(np.max(topics)+1)
    
    for topic in topic_titles :
    
        if topic in topic_doc_df['Topic'].unique():
            rslt_df = topic_doc_df[topic_doc_df['Topic'] == topic]
            docs = rslt_df['Document'].tolist()
            
            #print(f'Topic #{topic} \t# of documents : {len(docs)}')
            score = calc_topic_NPMI(docs, N)
        else:   
            score = 0
        TC_NPMI_scores.append(score)
      
    return TC_NPMI_scores, np.mean(TC_NPMI_scores)
    

def model_topics_in_batch(
    model_name,
    documents, 
    embeddings,
    labels,
    categories, 
    iters, 
    N,
    **model_params,
):
    '''
    Run specified topic model and parameters iters # of times and 
    build results into a single dataframe.
    '''
    result_df = pd.DataFrame()

    for i in range(iters):
    
        count_vectorizer = CountVectorizer()
        count_vectorizer.set_params(**model_params['CV_grid'])

        if model_name == 'LDA' :
        
            # build pipeline
            pipeline = Pipeline([
                ('CountVectorizer', count_vectorizer),
                ('TFIDF_transformer', TfidfTransformer()),
                ('LDA', LatentDirichletAllocation())
            ])
            pipeline.set_params(**model_params['LDA_grid'])
            
            # train, get topics for each document
            topic_probs = pipeline.fit_transform(documents)
            topics = np.argmax(topic_probs, axis = 1)
        
        elif model_name == 'BERTopic':
            topic_model = BERTopic(
                min_topic_size = model_params['min_topic_size'],
                
                # Used for BERTopic's built-in topic naming / top N word return
                # Not used for NPMI / Topic Coherence calculation
                vectorizer_model = count_vectorizer,
            )
                
            topics, probs = topic_model.fit_transform(documents, embeddings)
            
            max_ind = len(topic_model.get_topic_info())-1
            topics = [x if x != -1 else max_ind for x in topics]
                        
        else : 
            print(f'Model named : {model_name} invalid')
    
        # topic classification results
        results, results_dict = classification_matrix(
            labels, 
            categories,
            topics, 
        )
        
        # coherence score results
        TC_NPMI_scores, _ = model_coherence_score(topics, documents, N)
        
        i_result_df = pd.DataFrame(results, columns = categories )
        i_result_df['Run'] = i
        i_result_df['Topic'] = [x for x in range(len(TC_NPMI_scores))]
        i_result_df['NPMI Score'] = TC_NPMI_scores
        
        result_df = pd.concat([result_df, i_result_df])
   
    return result_df
    

def create_report(
    model_name,
    ID,
    iters,
    result_dir,
    
    result_df,
):
    '''
    Build report from results of model_topics_in_batch.
    '''
    fig, axes = plt.subplots(
        iters, 
        2, 
        squeeze = False, 
        gridspec_kw = {'hspace' : 0.25}, 
        **{'figsize' : (22, 14)}
    )
    plt.rcParams['font.size'] = 12
    fig.suptitle(f'{model_name} assignments and Topic Coherence scores. BBC data.', fontsize = 24)
    fig.text(0.5, 0.04, f'Topic Assignments', ha = 'center', fontsize = 20)
    fig.text(0.03, 0.5, f'{iters} runs', va = 'center', rotation = 'vertical', fontsize = 20)
    
    
    for i in range(iters):
    
        i_result_df = result_df.loc[result_df['Run'] == i]

        clf_df = i_result_df.loc[:, ~i_result_df.columns.isin(['Run', 'NPMI Score', 'Topic'])].transpose()
        # Topic classification plot
        sns.heatmap(   
            data = clf_df, 
            xticklabels = i_result_df['Topic'],
            annot = True,
            fmt = '.0f',
            robust = True,
            square = False,
            yticklabels = True,
            cbar = False,
            ax = axes[i, 0]
        )
        axes[i, 0].tick_params(axis = 'both', labelsize = 12)
        axes[i, 0].set_ylabel('#'+str(i), fontsize = 14)
        box = axes[i, 0].get_position()
        box.x0 = box.x0 + 0.045
        box.x1 = box.x1 + 0.045
        axes[i, 0].set_position(box)
                           
                    
        TC = i_result_df['NPMI Score'].mean()
        # Topic coherence plot
        sns.barplot( 
            data = i_result_df, 
            x = 'Topic', 
            y = 'NPMI Score',
            ax = axes[i, 1]
        )
        
        # Values over bars
        #axes[i, 1].bar_label()
        axes[i, 1].set_xlabel('Topic', fontsize = 14)
        axes[i, 1].set_ylabel('NPMI Score', fontsize = 14)
        axes[i, 1].tick_params(axis = 'both', labelsize = 12)        
        axes[i, 1].text( 0.05, 0.718, f'TC = {TC:.3f}', transform=axes[i][1].transAxes, fontsize = 14)
        box = axes[i, 1].get_position()
        box.x0 = box.x0 + 0.065
        box.x1 = box.x1 + 0.065
        axes[i, 1].set_position(box)
    '''
    plt.subplots_adjust(
        hspace = 0.4
    )
    '''
    fig.savefig(result_dir + 'results_'+str(ID)+'.png')
    print(f'ID created : {ID}')
    
    return