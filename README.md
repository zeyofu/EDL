# QuEL

This is code for paper Design Challenges in Low-resource Cross-lingual Entity Linking (EMNLP'20). [https://arxiv.org/abs/2005.00692](https://arxiv.org/abs/2005.00692)

We have a demo here [https://cogcomp.seas.upenn.edu/page/publication_view/911](https://cogcomp.seas.upenn.edu/page/publication_view/911)!

## How to Cite

A good research work is always accompanied by a thorough and faithful reference. If you use or extend our work, please cite the following paper:

```
```

# Run Model

### Clone Git Repository

Clone the repository from our github page (don't forget to star us!)
```bash
git clone https://github.com/zeyofu/EDL.git
```
## Install all the requirements:

```
pip install -r requirements.txt
```

## Prepare Data and MongoDB
#### Preprocessing Wikipedia for Candidate Generation
First, preprocess the Wikipedia dump for the languages you care about using the [wikidump_preprocessing](https://github.com/shyamupa/wikidump_preprocessing) code here.

#### Setting up Candidate Generation
The code for performing candidate generation is available at [wiki_candgen](https://github.com/shyamupa/wiki_candgen). It uses the resources generated using the [wikidump_preprocessing](https://github.com/shyamupa/wikidump_preprocessing) repo.


## Prepare Google ID
1. Get google_api_key following https://developers.google.com/custom-search/v1/overview. 
2. Get google_api_cx (search engine ID) from [https://programmablesearchengine.google.com/](https://programmablesearchengine.google.com/), on which set "Sites to search" as *.wikipedia.org, map.google.com. 
3. Use google_api_cx and google_api_key as parameters to run src/link_entity.py

## Pretrained Bert Model
Download pretrained Multilingual Bert or train own Bert model.


## Run the Model

```
python link_entity.py --kbdir ${kbdir} --lang ${lang} --year ${year}
```
where kbdir is directory of preprocessed wikipedia, lang is languagge, and year is wikipedia version (e.g. "20191020").


### Setting Up Resources
For faster processing we store the various maps (e.g. string to Wikipedia candidates) in a mongodb database collection. To start up the Mongo DB daemon, run: 
```bash
mongod --dbpath your_mongo_path
``` 
The `dbpath` argument is where mongodb creates the database and indexes. You can specify a different path, but you need to rebuild the indices below. 
Once the daemon is running, we can run the candidate generation script. 

