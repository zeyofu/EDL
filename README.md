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

## Pretrained Bert Model

## Run the Model


The `candgen_v2_one_file.py` script generates a list of candidate entities from the Lorelei KB for each NER mention (as identified by the NER_CONLL view). 

### Setting Up Resources
For faster processing we store the various maps (e.g. string to Wikipedia candidates, string to Lorelei KB candidates etc.) in a mongodb database collection. MongoDB stores various statistics (e.g. inlink counts for each Wikipedia page) and string-to-candidate indices that are used to compute candidates. To start up the Mongo DB daemon, run: 
```bash
mongod --dbpath /shared/bronte/upadhya3/tac2018/mongo_data
``` 
The `dbpath` argument is where mongodb creates the database and indexes. You can specify a different path, but you need to rebuild th indices below. 

Once the daemon is running, we can run the candidate generation script. 

### Candidate Generator
The candidate generation script `python_src/candgen_v2_one_file.py` takes a json serialized text annotation (containing the NER_CONLL view) and adds a CANDGEN view, which contains a dictionary from a candidate to its score.

To add candidates to a directory of JSON files containing the NER_CONLL view, run: 
```bash
./python_src/candgen_folder.sh <input directory> <output directory> <IL code> <number of processes>
```
The above script processes several files (controlled by `number of processes` argument) from the `input directory` at a time, and writes out the json serialized text annotation to the `output directory`. Good values of `number of processes` are <=15. 

### Probability Files
The `CandidateGenerator` class loads Wikipedia matches for a given string, each mapped to the probability of these matches. 

**IMPORTANT**: Probability files are required for this and are computed by running the makefile from the **wiki_processing** directory using `lang=CODE make all`, where `CODE` is the two-letter language code used by Wikipedia to identify the language. In the process of loading probability, the CandidateGenerator uses the full phrase to generate a wiki matches to probabilities map. It uses the words of the phrase to generate another wiki matches to probabilities map. It repeats the phrase matching and word matching process using the English version of the token. In total, four probability maps can be generated.

### Inlinks
The inlinks dictionary loads a map of Wikipedia page to number of inlinks into it. We use the number of inlinks to a page as a measure of its popularity.

### Title Normalization
The `TitleNormalizer` class is used to resolve ambiguities with multiple titles referring to the same page. These variations of titles appear in redirect pages. A redirect page has no content in itself, but sends the reader to another article in the same wiki. For example, the title `Detective_Sherlock_Holmes` is synonymous with the title `Sherlock_Holmes` and thus redirects to the latter page. 

The Candgen class loads the following indices: GeoNames (Knowledge Base corresponding to the given ilcode),  Wiki entries, Alternative Names, IL10 Transliteration, Cheap Translation Dictionary, Nicknames. 

### Interactive Mode
The interactive mode for candidate generation allows the user to insert a surface form and immediately see the query results in CLI. To run in interactive mode, run: 
```bash
python3 ./python_src/candgen_v2_one_file.py --ilcode <IL code> --lang <si or rw> --interactive
```
With the interactive flag, the user can enter a surface form as a query string. Here is a example interaction, (user inputs "kigali")

```
enter surface:kigali
INFO:root:===> query string:kigali len:6
INFO:root:#phrase cands altname 5 for kigali
INFO:root:#phrase cands geoname 11 for kigali
INFO:root:#direct cands (phrase): 14
INFO:root:# eids before prune, 14
cands found: 14
INFO:root:7659657 Kigali LOC H.STMI=intermittent stream H Rwanda 1.076 ext:Kigali
INFO:root:7599822 Kigali GPE P.PPLL=populated locality P Rwanda 1.076 ext:Kigali
INFO:root:202061 Kigali GPE P.PPLC=capital of a political entity P Rwanda 2.076 ext:Kigali
...
```

If the original surface form generates no candidates, the code backs up to various morphological forms that are used to re-query Cheap Translation and the KB. The code exits if any candidates are generated in each of the following steps, and proceeds to the next backup if no candidates are found.

The various morphological forms are the following (in order) :  
1. De-suffixed form
1. Suffixed form
3. Replaced suffix form
4. De-prefixed form 
6. Word level querying (if original surface form has more than one token)

The algorithm returns NIL if no candidates are generated from the above processes. Returns the highest-probability candidate if there exists candidates from the above processes.

### Non-Interactive Mode
If there is no interactive flag, the same computations are performed but added as CANDGEN view in the input json file. However, an additional filter is applied for candidates: 
only candidates whose KB NER label matches the ner prediction in the NER CONLL view, and whose feature code name is not blacklisted are outputted to the json file. 

### Running it all together
To run the end-2-end pipeline to get EDL output use the script `candgen_cluster.sh`. This will run the candidate generation, perform NIL unlinking if needed, then NIL clustering (either with fuzzy string match or exact string match), and finally converts the document text-annotation jsons to submission tab file.   
