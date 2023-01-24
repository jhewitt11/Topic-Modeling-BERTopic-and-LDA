# Topic Modeling

In this study we observe how both LDA and BERTopic perform modeling the BBC news dataset. 

## Contents
Dataset
Background
Methods
Results
Citations

## Dataset
2225 BBC news articles gathered from 2004-2005. 

5 different categories : 
- Sports
- Politics
- Entertainment
- Business
- Tech

## Background

#### Topic Modeling

#### Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) is one of the most common methods for topic modeling. It uses a bag-of-words approach meaning individual documents are represented only as a list of the frequencies of the words found in them.

This is an easy, reasonably effective way to vectorize or embed text but we of course lose a lot of information this way compared to more modern methods.

LDA assumes that within a set of documents there are a certain number of topics, that words can belong to one or multiple topics, and that documents can pertain to one or multiple topics. Dirichlet distributions characterize 
- dirichlet picture 

#### BERTopic

BERTopic is a method for topic modeling that uses BERT to vectorize documents and HDBSCAN & UMAP to group them into topics.  

## Methods

- raw text / categories to CSV
- raw CSV to clean CSV
- 

## Results

## References 

#### BERTopic paper :
@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}

#### BBC Dataset :
@inproceedings{greene06icml,
	Author = {Derek Greene and P\'{a}draig Cunningham},
	Booktitle = {Proc. 23rd International Conference on Machine learning (ICML'06)},
	Pages = {377--384},
	Publisher = {ACM Press},
	Title = {Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering},
	Year = {2006}}





