# Topic Modeling

## Contents
- [Introduction](#introduction)
- [Walkthrough](#walkthrough)
- [Dataset](#dataset)
- [Results](#results)
- [References](#references) 

## Introduction

Topic modeling is an unsupervised process where a model analyzes a collection of documents and creates topics by grouping similar documents together. This helps to answer very natural questions that arise when trying to understand large amounts of text. Questions like..

- What are these documents about? 
- How similar are these documents?
- Which other documents are similar to this one?

In this study we observe how two topic models, Latent Dirichlet Allocation (LDA) and BERTopic, perform modeling the BBC news dataset. 

#### why?
To make informed decisions on when to use models and how to tune them, we need quick/inexpensive methods to gauge their performance. Human judgement is the gold standard for judging relatedness in text/language but this approach doesn't scale well. Humans can only read and comprehend small amounts of text at a time.

#### how?
The first method we use to test our models is to look at how documents from a labelled dataset are distributed across the model's created topics. In the BBC dataset each document has one of 5 category labels..

- Sports
- Politics
- Entertainment
- Business
- Tech

The fewer documents from different categories a topic contains the better. For BERTopic models, the outlier topic is always the highest numbered topic. LDA models do not have an outlier topic. In the generated reports, the left graphic for each run is a matrix that looks like the one below. In this case, the model's topic 0 contains 327 documents labelled 'tech', 11 labelled 'politics', 7 labelled 'entertainment' and 14 labelled 'business'.

![alt text](./README_resources/heatmap.png)

The second method used to test the models is a calculated value. The NPMI score [-1, 1] of two words represents how often they occur within the same document. A score of -1 means they never appear together, 1 means they always appear together. The NPMI score of a topic is the average NPMI score of each pair of the most common `N` words. The average NPMI score across topics is the model's topic coherence score (TC). There are a few different variants of this calculation and links to more information are in the References section. The implementation used in this study can be found in `calc_topic_NPMI`. The bar graph on the right of the report shows each topic's NPMI score, with the model's TC score displayed in text.

![alt text](./README_resources/topicCoherence.png)


## Walkthrough

### run_study
This is the main script that creates `iters` number of topic models, evaluates each and builds a report of the results. 

In order for `run_study.py` to run properly, `read_in_data` needs a CSV that contains a document and label column. The document embeddings for the same data need to be read in as well.
~~~
read_directory = 'data/clean/'
result_directory = 'results/'

documents, labels, categories =  read_in_data(read_directory+'BBC_data_CLEAN.csv',   
    x_col = 1,  # column holding documents                              
    y_col = 2   # column holding labels
)

embeddings = joblib.load(read_directory + 'BBC_embeddings.z')
~~~

After the data is read in and parameters defined, the two functions below handle the rest of the work.

`model_topics_in_batch` trains the models, calculates topic coherence for each topic, and adds the results from each run to `result_df`.
~~~
result_df = model_topics_in_batch(
    model_name, 	# LDA or BERTopic
    documents, 		# cleaned document files
    embeddings, 	# embeddings for document files
    labels, 		# category for each document
    categories, 	# list of different categories in labels
    iters, 			# number of iterations to run
    N, 				# parameter for topic coherence, top N words are analyzed 
    **model_params	# parameters for LDA and BERTopic models
)
~~~
`create_report` creates a visual report from the results. The left matrix shows label distribution across topics. The right graph shows each topic's coherence score. The average of the topic's scores is the model's score. This value is displayed. 
~~~
create_report(
    model_name,         # LDA or BERTopic
    ID,                 # randomly generated ID
    iters,              # number of iterations
    result_directory,   # write location for report
    result_df,          # output from model_topics_in_batch
)
~~~ 
 ### build_dataset.py
The original BBC dataset is stored as individual files in labelled folders. This script compiles the data into a CSV and stores it in the same directory.

Set 'directory' to the location of the folders.
~~~
directory = 'data/raw/bbc/'

...

# Write out dataframe.
df.to_csv('data/raw/'+'BBC_data.csv', index_label = 'Index')
~~~

### clean_dataset
This script takes a dataset stored in a CSV file and prepares two versions that can be used by the models. This is not specific to the BBC data, the text and category columns to be processed are defined at the top by `TEXT_COL` and `CAT_COL`.


Both versions have null values dropped, duplicates dropped, and are balanced and cleaned via `undersample_dataframe` and `clean_text` functions in `tools.py`. The second version has `preprocess_text` applied as well.
~~~
...

# Undersampled and cleaned
data_df.to_csv(directory + 'clean/' + name + '_CLEAN.csv', index = False)

data_df[TEXT_COL] = data_df[TEXT_COL].apply(preprocess_text, args = [CUSTOM_STOPWORDS])

# Undersampled, cleaned and preprocessed
data_df.to_csv(directory + 'clean/' + name + '_CLEAN_P.csv', index = False)
~~~
### create_embeddings
The document embeddings created by a BERT model wouldn't meaningfully change from one iteration to another so we calculate them once here to be read in later.

~~~
#load sentence_model
model_name = "all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(model_name)

#embed/vectorize documents
embeddings = sentence_model.encode(documents)

#write out embeddings
joblib.dump(embeddings, read_directory + 'BBC_embeddings.z')
~~~

## Dataset
2225 BBC news articles gathered from 2004-2005. 

5 different categories : 
- Sports
- Politics
- Entertainment
- Business
- Tech

#### BBC Data document counts
|Category|Original|After cleaning and balancing|
|--------|--------|--------------|
|Sports|511|347|
|Politics|417|347|
|Entertainment|386|347|
|Business|510|347|
|Tech|401|347|


## Results

The results below are from a small sample, each model was run 8 times with the parameters below. 

~~~
BERTopic:
	# default parameters
	model_name = "all-MiniLM-L6-v2"
	min_topic_size = 10

LDA:
	CV_grid = {
		'max_df': 0.85, 
		'min_df': 0.1, 
		'ngram_range': (1, 1), 
	}
	LDA_grid = {    
		'LDA__n_components': 15, 
		'LDA__doc_topic_prior': 0.5, 
		'LDA__topic_word_prior': 0.5,
		'LDA__max_iter' : 100,
	}
	
General parameters:
	iters = 8
	N = 10
~~~

### BERTopic Results
![alt text](./README_resources/BERTopic.png)


### LDA Results
![alt text](./README_resources/LDA.png)

#### Topic Coherence Scores
| |BERTopic|LDA|
|--|--|--|
|Max|0.344|0.168|
|Min|0.213|0.114|
|Average|0.254|0.142|
|Standard Deviation|0.0523|0.0171|

BERTopic consistently builds topics that hold documents from a single category with only a few outliers. The topics generated by LDA are messier. This makes sense and can be explained by BERTopic’s more sophisticated, semantically rich document vectors as well as it’s outlier topic which holds hard to place documents.

BERTopic’s Topic Coherence scores were also consistently higher than the LDA models. `N` defines how many of the most frequently occuring words in a topic are used to calculate a topic’s NPMI score. This score is not weighted by the number of documents in a topic, and TC is the average of NPMI scores. 

Large topics, no matter the category consistently had a low NPMI score. LDA’s highest scoring topics were around 0.4 – 0.5. BERTopic regularly made topics that scored above 0.5, with a few topics scoring the maximum, 1.

## References 

#### BBC Data :
http://mlg.ucd.ie/datasets/bbc.html

@inproceedings{
greene06icml,
	Author = {Derek Greene and P\'{a}draig Cunningham},
	Booktitle = {Proc. 23rd International Conference on Machine learning (ICML'06)},
	Pages = {377--384},
	Publisher = {ACM Press},
	Title = {Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering},
	Year = {2006}
	}

#### BERTopic Paper :
https://arxiv.org/abs/2203.05794

@article{
grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}

#### Topic Coherence Papers :
https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf

@article{
	bouma2009,
	title = {Normalized (pointwise) mutual information in collocation extraction},
	author = {Gerlof Bouma}
	journal = {Proceedings of GSCL},
	year = {2009}
}

https://aclanthology.org/W13-0102.pdf

@inproceedings{aletras-stevenson-2013-evaluating,
    title = "Evaluating Topic Coherence Using Distributional Semantics",
    author = "Aletras, Nikolaos  and Stevenson, Mark",
    booktitle = "Proceedings of the 10th International Conference on Computational Semantics ({IWCS} 2013) {--} Long Papers",
    month = mar,
    year = "2013",
    address = "Potsdam, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W13-0102",
    pages = "13--22",
}


