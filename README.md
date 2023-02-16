This repository contains the code for the second extension of KRED using the LinkedIn Tech Job dataset and Stackovorflow Developer survey. 
The LikedIn Tech Job dataset is a collection of tech job offers scraped from LinkedIn in 2021.
Stackoverflow User survey is the collection of answers to the annual survey given by stackoverflow to its users.

## Introduction 
In this extension is exploit a way of trasfer learning in which, given a collection of job offers and a syntethic click log generated starting from information about some hypotethical user, they can be performed User2Item, Item2Item and Category Classification.

## Data Description
![](./framework.PNG)

We used [LinkedIn Tech Job Dataset](https://github.com/Mlawrence95/LinkedIn-Tech-Job-Data) and [StackOverflow Developer Survey Dataset](https://info.stackoverflowsolutions.com/rs/719-EMH-566/images/stack-overflow-developer-survey-2020.zip).
LinkedIn Tech Job Dataset contains tech job offers scraped from LinkedIn in 2021: 75 MB - 10.000 job offers.
StackOverflow Developer Survey Dataset contains answers to the annual survey given by StackOverflow to its users: 26 MB - 60.000 users. Specifaclly, we sampled 10.000 users.

## Model Description
From LinkedIn Tech Job Dataset we selected useful columns to emulate MIND structure and we used a pipeline of *NLTK* functions for entity linking in order to links entities to the corresponding entities in WikiData.

To represent each entity and relationship, we query the WikiData portal using the SPARQL protocol and RDF query language to obtain the title and description of each entity.
We then use SentenceTransformer with the all-mpnet-base-v2 model to obtain the corresponding embedding, which is a 768-dimensional dense vector space used to represent each embedding. The same procedure is used for relationship encoding. However, due to the large space requirement of loading the entire knowledge graph into memory, we load the embeddings into memory in the form of tensors and apply PCA to reduce the size to 100.

## Running the code

