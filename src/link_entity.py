import argparse, copy, json, os, sys, re
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
logging.basicConfig(level=logging.INFO)
import operator
from nltk.tokenize import MWETokenizer, sent_tokenize
import torch
import pymongo
from tqdm import tqdm
from utils.google_search import query2enwiki, query2gmap_api, or2hindi
from utils.language_utils import correct_surface, remove_suffix
from utils.mongo_backed_dict import MongoBackedDict
from wiki_kb.candidate_gen_v2 import CandidateGenerator
from wiki_kb.title_normalizer_v2 import TitleNormalizer
import utils.constants as K
import wikipediaapi

en_wiki = wikipediaapi.Wikipedia('en')
logging.basicConfig(format=':%(levelname)s: %(message)s', level=logging.INFO)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def mask_sents(surface, text):
    sents = sent_tokenize(text)
    masked_sents = []
    for sent in sents:
        pre_pos = 0
        while surface in sent[pre_pos:]:
           low = pre_pos + sent[pre_pos:].index(surface)
           high = low + len(surface)
           masked_sents.append('[CLS] ' + sent[:low] + ' [MASK] ' + sent[high:] + ' [SEP]')
           pre_pos = high + 1
    return masked_sents


def s2maskedvec(masked_sents):
    vecs = []
    for sent in masked_sents:
        tokenized_text = args.tokenizer.tokenize(sent)
        pos = tokenized_text.index('[MASK]')
        # Convert token to vocabulary indices
        indexed_tokens = args.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor)
            vecs.append(encoded_layers[11][0].numpy()[pos])
    m_vec = np.mean(vecs, axis=0)
    return m_vec

def get_wiki_summary(title):
    try:
        page_py = en_wiki.page(title)
        smr = page_py.summary
    except Exception as e:
        smr = ''
    return smr


class CandGen:
    def __init__(self, lang=None, year=None,
                 wiki_cg=None, google_api_cx=None, google_api_key=None):
        self.lang = lang
        self.year = year
        self.wiki_cg = wiki_cg
        self.en_normalizer = TitleNormalizer(lang="en")
        self.google_api_cx = google_api_cx
        self.google_api_key = google_api_key

    def load_kb(self, kbdir):
        self.en_t2id = MongoBackedDict(dbname=f"en_t2id")
        self.en_id2t = MongoBackedDict(dbname=f"en_id2t")
        en_id2t_filepath = os.path.join(kbdir, "enwiki", "idmap", f'enwiki-{self.year}.id2t')
        self.fr2entitles = MongoBackedDict(dbname=f"{self.lang}2entitles")
        fr2entitles_filepath = os.path.join(kbdir, f'{self.lang}wiki', 'idmap', f'fr2entitles')
        self.t2id = MongoBackedDict(dbname=f"{self.lang}_t2id")

        if self.en_t2id.size() == 0 or self.en_id2t.size() == 0:
            logging.info(f'Loading en t2id and id2t...')
            self.en_t2id.drop_collection()
            self.en_id2t.drop_collection()
            en_id2t = []
            ent2id = defaultdict(list)
            for line in tqdm(open(en_id2t_filepath)):
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    logging.info("bad line %s", line)
                    continue
                page_id, page_title, is_redirect = parts
                key = page_title.replace('_', ' ').lower()
                ent2id[key].append(page_id)
                en_id2t.append({'key': page_id, 'value':
                    {'page_id': page_id, 'name':page_title, 'searchname': key}, 'redirect':is_redirect})
            ent2id_list = []
            for k, v in ent2id.items():
                ent2id_list.append({'key': k, 'value': v})
            logging.info("inserting %d entries into english t2id", len(ent2id_list))
            self.en_t2id.cll.insert_many(ent2id_list)
            self.en_t2id.cll.create_index([("key", pymongo.HASHED)])

            logging.info("inserting %d entries into english id2t", len(en_id2t))
            self.en_id2t.cll.insert_many(en_id2t)
            self.en_id2t.cll.create_index([("key", pymongo.HASHED)])

        if self.fr2entitles.size() == 0:
            logging.info(f'Loading fr2entitles and {self.lang} t2id...')
            fr2en = []
            t2id = []
            f = open(fr2entitles_filepath)
            for idx, l in enumerate(f):
                parts = l.strip().split("\t")
                if len(parts) != 2:
                    logging.info("error on line %d %s", idx, parts)
                    continue
                frtitle, entitle = parts
                key = frtitle.replace('_', ' ').lower()
                enkey = entitle.replace('_', ' ').lower()
                fr2en.append({"key": key, "value": {'frtitle': frtitle, 'entitle': entitle, 'enkey':enkey}})
                t2id.append({"key": key, "value": self.en_t2id[enkey]})
            logging.info(f"inserting %d entries into {self.lang}2entitles", len(fr2en))
            self.fr2entitles.cll.insert_many(fr2en)
            self.fr2entitles.cll.create_index([("key", pymongo.HASHED)])
            logging.info(f"inserting %d entries into {self.lang} t2id", len(t2id))
            self.t2id.cll.insert_many(t2id)
            self.t2id.cll.create_index([("key", pymongo.HASHED)])

    def clean_query(self, query_str):
        if self.lang == 'rw':
            if 'North' == query_str[-5:]:
                query_str = query_str[:-5] + ' ' + query_str[-5:]

        if query_str[-1] == ",":
            query_str = query_str[:-1]
        f = re.compile('(#|\(|\)|@)')
        query_str = f.sub(' ', query_str)
        query_str = re.sub('\s+', ' ', query_str).strip()
        return query_str.lower()

    def init_l2s_map(self, eids, args = None):

        l2s_map = {}
        for eid in eids:
            flag = False
            title = (self.en_id2t[eid]["name"] if eid in self.en_id2t else eid.replace(' ', '_')).lower()
            if 'category:' in title:
                continue
            key = "|".join([eid, title])
            if not flag and not key in l2s_map:
                l2s_map[key] = 100
        return l2s_map

    def cross_check_score(self, l2s_map, eids):
        freq = dict(Counter(eids))
        for cand, v in l2s_map.items():
            cand_eid = cand.split("|")[0]
            l2s_map[cand] = l2s_map[cand] * (3 ** freq[cand_eid]) if cand_eid in eids else l2s_map[cand] * 0.1
        return l2s_map

    def bert_score(self, l2smap, query_emb, l2s_map, args):
        cand2sim = {}
        max_cand = None
        max_sim = -1000
        no_sum = 0
        not_in_sum = 0
        for cand in l2smap:
            cand_title = cand.split("|")[0]
            # request summary
            if cand_title in args.eid2wikisummary:
                summary = args.eid2wikisummary[cand_title]
            else:
                summary = get_wiki_summary(cand_title)
                args.eid2wikisummary.cll.insert_one({"key": cand_title, "value": summary})
            summary = summary.lower()
            cand_name = cand_title.lower()
            if summary == '':
                no_sum += 1
            # bert
            else:
                if cand_name in summary:
                    cand_emb = s2maskedvec(mask_sents(cand_name, summary))
                    sim = cosine_similarity([cand_emb], [query_emb])[0][0]
                    cand2sim[cand] = sim
                    if sim > max_sim:
                        max_sim = sim
                        max_cand = cand
                else:
                    not_in_sum += 1
                    continue
        logging.info(f"{no_sum} / {len(l2s_map)} dont have summary, {not_in_sum} / {len(l2s_map) - no_sum} not in summary")
        if len(cand2sim) > 1:
            l2s_map[max_cand] *= 3 * (len(l2s_map) - no_sum - not_in_sum) / len(l2s_map)
        return l2s_map

    def get_context(self, query_str, text, k = 10):
        if query_str in text:
            tokenizer = MWETokenizer()
            query_str_tokens = tuple(query_str.split())
            query_str_dashed = "_".join(query_str_tokens)
            tokenizer.add_mwe(query_str_tokens)
            text_token = tokenizer.tokenize(text.split())
            try:
                t_start = text_token.index(query_str_dashed)
            except:
                return None, None, None
            t_end = t_start + 1
            start_index = max(t_start - k, 0)
            end_index = min(t_end + k, len(text_token))
            text_token_query = text_token[start_index:t_start] + text_token[t_end + 1:end_index]
            context = " ".join(text_token_query)
            context_mention = text_token[start_index:t_start] + [query_str] + text_token[t_end + 1:end_index]
            context_mention = " ".join(context_mention)
            return context, text_token_query, context_mention
        else:
            logging.info('error, query not in text')
            return None, None, None

    def get_l2s_map(self, eids, eids_google, eids_pivot, eids_google_maps, eids_wikicg, eids_total, ner_type, query_str, text, args):
        if args.wikidata:
            l2s_map = self.init_l2s_map(eids_wikicg + eids + eids_google + eids_pivot + eids_google_maps, args=args)
        else:
            l2s_map = self.init_l2s_map(eids_total, args=args)

        # check if generated cadidates
        if len(l2s_map) <= 1:
            return l2s_map
        if ner_type in ['GPE', 'LOC']:
            l2s_map = self.cross_check_score(l2s_map, eids_google + eids_google_maps)
        if args.wikidata:
            l2s_map = self.cross_check_score(l2s_map, eids_wikicg)
        else:
            l2s_map = self.cross_check_score(l2s_map, eids + eids_google + eids_google_maps)
        #update score
        for cand in l2s_map.copy().keys():
            cand_eid, cand_text = cand.split("|")
            score = 1
            if self.lang == 'rw':
                if 'rwanda' in cand_text.split('_'):
                    score *= 3
            l2s_map[cand] *= score
        logging.info("Processed looping candidates")

        # get context:
        if args.bert:
            context, context_tokens, context_mention = self.get_context(query_str, text, k=10)
            # check context bert
            if context is not None:
                logging.info("Processing candidates bert")
                query_emb = s2maskedvec(mask_sents(query_str, context_mention))
                l2s_map = self.bert_score(l2s_map, query_emb, l2s_map, args)

        # Normalize
        sum_s = sum(list(l2s_map.values()))
        for can, s in l2s_map.items():
            l2s_map[can] = s/sum_s
        return l2s_map

    def correct_surf(self,token):
        region_list = ["district of", "district", "city of", "state of", "province of", "division", "city", "valley","province"]
        token = token.lower()
        for i in region_list:
            token = token.replace(i, "").strip()
        return token

    def get_maxes_l2s_map(self, l2s_map):
        # pick max
        if len(l2s_map) == 0:
            max_cand, max_score = "NIL", 1.0
        else:
            maxes_l2s_map = {cand: score for cand, score in l2s_map.items() if score == max(l2s_map.values())}
            max_cand = list(maxes_l2s_map.keys())[0]
            max_score = l2s_map[max_cand]
        return max_cand, max_score

    def compute_hits_for_ta(self, input_json, outfile=None, args=None):
        output_json = copy.deepcopy(input_json)
        ner_entities = output_json["NER"]
        text = output_json['text']

        for idx, cons in enumerate(ner_entities):
            orig_query_str = cons["tokens"]
            ner_type = cons["ner_type"]

            query_str = self.clean_query(orig_query_str)
            eids, eids_google, eids_pivot, eids_google_maps, eids_wikicg = self.get_all_candidates(orig_query_str, query_str, ner_type=ner_type, args=args)
            eids_total = eids + eids_google + eids_pivot + eids_google_maps + eids_wikicg

            logging.info("got %d candidates for query:%s", len(set(eids_total)), orig_query_str)

            l2s_map = self.get_l2s_map(eids, eids_google, eids_pivot, eids_google_maps, eids_wikicg, eids_total, ner_type=ner_type, query_str=orig_query_str, text=text, args=args)
            l2s_map = dict((x, y) for x, y in sorted(l2s_map.items(), key=operator.itemgetter(1), reverse=True))
            logging.info(f"got {len(l2s_map)} candidates after ranking for {orig_query_str}: {l2s_map}")
            max_cand, max_score = self.get_maxes_l2s_map(l2s_map)

            if len(l2s_map) > 0:
                # do not send empty label2scoremaps!
                cons["labelScoreMap"] = l2s_map
            cons["label"] = max_cand
            cons["score"] = max_score
        if outfile is not None:
            json.dump(output_json, open(outfile, 'w'))
        return output_json

    def get_all_candidates(self, orig_query_str, query_str, ner_type=None, args=None):
        # SKB+SG
        desuf_query_str = remove_suffix(query_str, self.lang)
        dot_query_str_list = correct_surface(query_str, self.lang)
        desuf_dot_query_str_list = correct_surface(desuf_query_str, self.lang)
        # SKB suffix dot suffix+dot
        eids = self._exact_match_kb(query_str, args)
        eids += self._exact_match_kb(desuf_query_str, args)
        for i in dot_query_str_list:
            eids += self._exact_match_kb(i, args)
        for i in desuf_dot_query_str_list:
            eids += self._exact_match_kb(i, args)
        eids = list(set(eids))

        if args.wikicg:
            wiki_titles, wids, wid_cprobs = self._extract_ptm_cands(self.wiki_cg.get_candidates(surface=orig_query_str))
            eids_wikicg = []
            for w in wids:
                if self.en_id2t[w]["name"] not in eids_wikicg:
                    eids_wikicg.append(self.en_id2t[w]["name"])
            if args.google + args.google_map == 0:
                eids = []
        else:
            eids_wikicg = []
            
        eids_google = []
        if args.google and ((not args.wikidata) or (args.wikidata and len(eids_wikicg) == 0)):
            # SG suffix dot suffix+dot
            eids_google, g_wikititles = self._get_candidates_google(query_str, top_num=args.google_top)

            # if not eids_google:
            for i in [desuf_query_str] + dot_query_str_list + desuf_dot_query_str_list:
                es, ts = self._get_candidates_google(i, top_num=args.google_top)
                for e in es:
                    if e not in eids_google:
                        eids_google.append(e)
                for t in ts:
                    if t not in g_wikititles:
                        g_wikititles.append(t)
        eids_google = list(set(eids_google))
        logging.info("got %d candidates for query:%s from google", len(eids_google), query_str)

        eids_google_maps = []
        if ner_type in ['GPE', 'LOC'] and args.google_map and ((not args.wikidata) or (args.wikidata and len(eids_wikicg) == 0)):
            google_map_name = query2gmap_api(query_str, self.lang, self.google_api_key)
            eids_google_maps += self._exact_match_kb(google_map_name, args)
            eids_google_maps += self._get_candidates_google(google_map_name, lang='en', top_num=args.google_top)[0]
            google_map_name_suf = query2gmap_api(desuf_query_str, self.lang, self.google_api_key)
            google_map_name_dot = [query2gmap_api(k, self.lang, self.google_api_key) for k in dot_query_str_list]
            google_map_name_suf_dot = [query2gmap_api(k, self.lang, self.google_api_key) for k in desuf_dot_query_str_list]
            eids_google_maps += self._exact_match_kb(google_map_name_suf, args)
            eids_google_maps += self._get_candidates_google(google_map_name_suf, lang='en', top_num=args.google_top)[0]
            eids_google_maps += [h for k in google_map_name_dot for h in self._exact_match_kb(k, args)]
            eids_google_maps += [h for k in google_map_name_dot for h in self._get_candidates_google(k, lang='en', top_num=args.google_top)[0]]
            eids_google_maps += [h for k in google_map_name_suf_dot for h in self._exact_match_kb(k, args)]
            eids_google_maps += [h for k in google_map_name_suf_dot for h in self._get_candidates_google(k, lang='en', top_num=args.google_top)[0]]
        eids_google_maps = list(set(eids_google_maps))
        logging.info("got %d candidates for query:%s from google map", len(set(eids_google_maps)), query_str)


        eids_pivot = []
        if args.pivoting:
            if self.lang == 'or':
                if len(eids) + len(eids_google) + len(eids_google_maps) == 0:
                    orgin2hin = or2hindi(query_str)
                    eids_pivot += self._get_candidates_google(orgin2hin, lang='hi', top_num=args.google_top)[0]
                    suf2hin = or2hindi(desuf_query_str)
                    dot2hin = [or2hindi(k) for k in dot_query_str_list]
                    suf_dot2hin = [or2hindi(k) for k in desuf_dot_query_str_list]
                    eids_pivot += self._get_candidates_google(suf2hin, lang='hi', top_num=args.google_top)[0]
                    eids_pivot += [h for k in dot2hin for h in self._get_candidates_google(k, lang='hi', top_num=args.google_top)[0]]
                    eids_pivot += [h for k in suf_dot2hin for h in self._get_candidates_google(k, lang='hi', top_num=args.google_top)[0]]
            else:
                if len(eids) + len(eids_google) + len(eids_google_maps) == 0:
                    eids_pivot += self._get_candidates_google(query_str + 'wiki', top_num=args.google_top, include_all_lang=True)[0]
            eids_pivot = list(set(eids_pivot))
        logging.info("got %d candidates for query:%s from pivoting", len(eids_pivot), query_str)

        return eids, eids_google, eids_pivot, eids_google_maps, eids_wikicg

    def _get_candidates_google(self, surface, top_num=1, lang=None, include_all_lang=False):
        eids = []
        wikititles = []
        if surface is None or len(surface) < 2:
            return eids, wikititles
        if lang is None:
            lang = self.lang
        en_surfaces = query2enwiki(surface, lang, self.google_api_key, self.google_api_cx, include_all_lang=include_all_lang)[:top_num]
        for en in en_surfaces:
            if en not in wikititles:
                wikititles.append(en)
        eids = wikititles
        return eids, wikititles

    def _extract_ptm_cands(self, cands):
        wiki_titles, wids, wid_cprobs = [], [], []
        for cand in cands:
            wikititle, p_t_given_s, p_s_given_t = cand.en_title, cand.p_t_given_s, cand.p_s_given_t
            nrm_title = self.en_normalizer.normalize(wikititle)
            if nrm_title == K.NULL_TITLE:
                logging.info("bad cand %s nrm=%s", wikititle, nrm_title)
                continue
            wiki_id = self.en_normalizer.title2id[nrm_title]
            if wiki_id is None:
                wiki_id = self.en_normalizer.title2id[wikititle]
                if wiki_id is None:
                    continue
                wiki_titles.append(wikititle)
                wids.append(wiki_id)
                wid_cprobs.append(p_t_given_s)
                continue
            wiki_titles.append(nrm_title)
            wids.append(wiki_id)
            wid_cprobs.append(p_t_given_s)
        return wiki_titles, wids, wid_cprobs

    def _exact_match_kb(self, surface):
        eids = []
        if surface is None:
            return eids
        if len(surface) < 2:
            return []

        # Exact Match
        eids += self.get_phrase_cands(surface)
        return eids

    def get_phrase_cands(self, surf):
        surf = surf.lower()
        ans = []
        if surf in self.t2id:
            cands = self.t2id[surf]
            ans += cands
        if surf in self.en_t2id:
            cands = self.en_t2id[surf]
            if len(cands) > 0:
                cand = self.en_id2t[cands[0]]["name"]
                ans += [cand]
        return ans


if __name__ == '__main__':
    # For demo
    PARSER = argparse.ArgumentParser(description='Short sample app')
    PARSER.add_argument('--kbdir', default="../data/wiki_outdir", type=str, help='dir of preprocessed wiki')
    PARSER.add_argument('--lang', default='ilo', type=str)
    PARSER.add_argument('--year', default="20191020")
    PARSER.add_argument('--input_file', default='../data/input/example_illocano', help='json input with text and ner entities')
    PARSER.add_argument('--wikidata', default=0, type=int, help='if data source is wikipedia')
    PARSER.add_argument('--wikicg', default=1, type=int, help='use P(e|m)')
    # For Cand gen
    PARSER.add_argument('--google', default=1, type=int)
    PARSER.add_argument('--google_top', default=5, type=int)
    PARSER.add_argument('--google_map', default=1, type=int)
    PARSER.add_argument('--google_api_cx', default='', type=str, help='https://programmablesearchengine.google.com/')
    PARSER.add_argument('--google_api_key', default='', type=str, help='https://developers.google.com/custom-search/v1/overview')
    PARSER.add_argument('--pivoting', default=1, type=int)
    # For Ranking
    PARSER.add_argument('--bert', default=0, type=int)
    PARSER.add_argument('--bert_model_path', default='', type=str, help='use downloaded pretrained multilingual bert')
    args = PARSER.parse_args()
    logging.info(args)
    logging.disable(logging.INFO)

    if not args.lang:
        sys.exit('No language specified.')
    if args.google and (not args.google_api_key or not args.google_api_cx):
        sys.exit('No google api key and cx')

    # load pro-processed wiki
    if args.wikicg:
        wiki_cg = CandidateGenerator(K=6, kbfile=None, lang=args.lang, use_eng=False, fallback=True)
        wiki_cg.load_probs("data/{}wiki/probmap/{}wiki-{}".format(args.lang, args.lang, args.year))
    else:
        wiki_cg = None

    # load bert
    if args.bert:
        args.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained(args.bert_model_path, from_tf=False)
    lang = args.lang

    cg = CandGen(lang=args.lang, year=args.year, wiki_cg=wiki_cg, google_api_cx=google_api_cx, google_api_key=google_api_key)
    cg.load_kb(args.kbdir)

    # Mongodb
    args.eid2wikisummary = MongoBackedDict(dbname="eid2wikisummary")

    #load data and link
    input_json = json.load(open(args.input_file))
    linking_results = cg.compute_hits_for_ta(input_json, outfile=None, args=args)
    print(linking_results)
