import joblib
import pandas as pd

import bertopic
from bertopic import BERTopic

import matplotlib.pyplot as plt
import seaborn as sns

from tools import read_in_data
from tools import BERTopic_topic_report
from tools import classification_matrix
from tools import model_topics_in_batch
from tools import calc_topic_coherence

directory = 'data/clean/'

embeddings = joblib.load(directory + 'BBC_embeddings.z')
documents, labels, categories =  read_in_data(directory+'BBC_data_CLEAN_P.csv', x_col = 1, y_col = 2)


# Give it preprocessed text for easier to understand results from C-Tf-idf
topic_model = BERTopic(nr_topics = 5)
topics, probs = topic_model.fit_transform(documents, embeddings)


all_topics = topic_model.get_topics()


freq_df = topic_model.get_topic_freq()
BERTopic_topic_report(freq_df, all_topics, j = 5)        
'''
results, results_dict = classification_matrix(labels, 
                                                categories,
                                                topics, 
                                                )
                                                
result_df = pd.DataFrame(results, columns = categories ).transpose()
'''


model_topics_in_batch(documents,
                        labels,
                        embeddings,
                        categories,
                        iters = 2,
                        set_topic_num = False,
                        result_dir = 'results/')
                        
                        
                        

'''   


# Topic coherence calculation

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(documents, embeddings)


N = 3
TC_NPMI_scores = []
str_topic_titles = []

'''
sample_topic_titles = [x for x in range(6)]
sample_docs = ['hello this this is this document one', 
        'this is my my second attempt doggo', 
        'okay Im over',
        'hello one flew over the cuckoos nest',
        'attempt to steal document',
        'nest is okay but doggo like my bed'
    ]
'''

BT_docs_df = topic_model.get_document_info(documents)
topic_titles = BT_docs_df['Topic'].unique()
topic_titles.sort()

for topic in topic_titles :
  rslt_df = BT_docs_df[BT_docs_df['Topic'] == topic]
  docs = rslt_df['Document'].tolist()
  score = calc_topic_coherence(docs, N)
  
  TC_NPMI_scores.append(score)
  str_topic_titles.append(str(topic))
  

coherence_score_df = {'Topic' : str_topic_titles,
                        'Coherence Score' : TC_NPMI_scores
                        }
sns.barplot( data = coherence_score_df, x = 'Topic', y = 'Coherence Score')

plt.show()

'''