# QuEL

This is code for paper Design Challenges in Low-resource Cross-lingual Entity Linking (EMNLP'20). [https://arxiv.org/abs/2005.00692](https://arxiv.org/abs/2005.00692)

We have a demo here [https://cogcomp.seas.upenn.edu/page/publication_view/911](https://cogcomp.seas.upenn.edu/page/publication_view/911)!

## How to Cite

A good research work is always accompanied by a thorough and faithful reference. If you use or extend our work, please cite the following paper:

```
@inproceedings{FSYZR20,
    author = {Xingyu Fu and Weijia Shi and Xiaodong Yu and Zian Zhao and Dan Roth},
    title = {{Design Challenges in Low-resource Cross-lingual Entity Linking}},
    booktitle = {Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2020},
    url = "https://cogcomp.seas.upenn.edu/papers/FSYZR20.pdf",
    funding = {LORELEI},
}
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
### Setting up mongo
For faster processing we store the various maps (e.g. string to Wikipedia candidates, string to Lorelei KB candidates etc.) in a mongodb database collection. MongoDB stores various statistics and string-to-candidate indices that are used to compute candidates. To start up the Mongo DB daemon, run: 
```bash
mongod --dbpath your_mongo_path
``` 
The `dbpath` argument is where mongodb creates the database and indexes. You can specify a different path, but you need to rebuild th indices below. 

Note the mongo should be run in the port 27017. 

### Preprocessing Wikipedia for Candidate Generation
First, preprocess the Wikipedia dump for the languages you care about using the [wikidump_preprocessing](https://github.com/shyamupa/wikidump_preprocessing) code here. Preprocessing wikis into a folder outwiki consisting of different language wikipedia folder. For example, the layout of outwiki folder should be:
```
– outwiki:\
   &emsp;  – enwiki\
    &emsp; – eswiki\
    &emsp; – zhwiki \
    &emsp; ....
```
### load data to Mongo
Set the abs_path in `utils/misc_utils.py` to `outwiki`. When you first time run the `link_entity.py`, data will be automatically loaded to mongo. (It may take a long time)


## Prepare Google ID
1. Get `google_api_key` following https://developers.google.com/custom-search/v1/overview. 
2. Get `google_api_cx` (search engine ID) from [https://programmablesearchengine.google.com/](https://programmablesearchengine.google.com/), on which set "Sites to search" as `*.wikipedia.org`, `map.google.com`. 
3. Use `google_api_cx` and `google_api_key` as parameters to run src/link_entity.py

## Pretrained Bert Model
Download pretrained Multilingual Bert or train own Bert model.


## Run the Model

```
cd src
python link_entity.py --kbdir ${kbdir} --lang ${lang} --year ${year}
```
where `kbdir` is directory of preprocessed wikipedia `outwiki`, `lang` is [language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) used in Preprocessing Wikipedia, and `year` is downloaded wikipedia version (e.g. "20191020").



