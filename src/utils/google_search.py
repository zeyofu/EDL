import sys
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
from utils.mongo_backed_dict import MongoBackedDict
import googlemaps


replace = False

def isEnglish(query):
    if ORIYA_BASE <= ord(query[int(len(query)/2)]) <= ORIYA_MAX:
        return False
    if ORIYA_BASE <= ord(query[int(len(query)/3)]) <= ORIYA_MAX:
        return False
    if ORIYA_BASE <= ord(query[int(len(query)/1.5)]) <= ORIYA_MAX:
        return False
    return True

def query2enwiki(query, lang, api_key, api_cx, include_all_lang=False):
    urls, en_entities = None, None
    if lang == 'ilo':
        query += ' philippines'
    elif lang == 'or' and isEnglish(query):
        query += ' india'
    mention2gentities = MongoBackedDict(dbname=f"mention2gentities_{lang}")
    mention2gurls = MongoBackedDict(dbname=f"mention2gurls_{lang}")
    if query in mention2gentities and query in mention2gurls:
        en_entities = mention2gentities[query]
        urls = mention2gurls[query]
        assert urls is not None
        assert en_entities is not None
    else:
        if query in mention2gentities:
            mention2gentities.cll.delete_one({'key': query})
        elif query in mention2gurls:
            mention2gurls.cll.delete_one({'key': query})

        PARAMS = {'key': api_key, 'q': query, 'cx': api_cx, 'num':5}
        requests.adapters.DEFAULT_RETRIES = 5
        s = requests.session()
        s.keep_alive = False
        r = requests.get(url="https://www.googleapis.com/customsearch/v1/siterestrict?", params=PARAMS)
        return_data = r.json()
        if 'items' in return_data:
            en_entities = []
            urlss = return_data['items']
            urls = [u['link'] for u in urlss]
            for url in urls:
                enwiki_url = None
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                if domain == "en" + '.wikipedia.org':
                    enwiki_url = url
                elif domain == lang + '.wikipedia.org':
                    res = requests.get(url)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    for li in soup.find_all('li', {'class': 'interlanguage-link interwiki-en'}):
                        enwiki_url = li.find("a")['href']
                    if enwiki_url is None:
                        for li in soup.select('li.interlanguage-link > a'):
                            link = li.get('href')
                            if 'en.wikipedia.org' in link:
                                enwiki_url = link
                                break
                elif include_all_lang is True and '.wikipedia.org' in domain:
                    res = requests.get(url)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    for li in soup.find_all('li', {'class': 'interlanguage-link interwiki-en'}):
                        enwiki_url = li.find("a")['href']
                    if enwiki_url is None:
                        for li in soup.select('li.interlanguage-link > a'):
                            link = li.get('href')
                            if 'en.wikipedia.org' in link:
                                enwiki_url = link
                                break
                if enwiki_url is not None:
                    en_res = requests.get(enwiki_url, verify=True)
                    en_soup = BeautifulSoup(en_res.text, 'html.parser')
                    entity = en_soup.title.string[:-12]
                    if not entity in en_entities:
                        en_entities.append(entity)
        elif return_data['queries']['request'][0]['totalResults'] == '0':
            urls = []
            en_entities = []

        assert urls is not None
        assert en_entities is not None
        mention2gurls.cll.insert_one({"key": query, "value": urls})
        mention2gentities.cll.insert_one({"key": query, "value": en_entities})

    return en_entities




def query2gmap_api(text, lang, api_key):
    gmaps = googlemaps.Client(key=api_key)
    if lang == 'or' and isEnglish(text):
        text += ' india'
    mention2gmap_entity = MongoBackedDict(dbname=f"mention2gmap_entity_{lang}")
    if text in mention2gmap_entity:
        en_entities = mention2gmap_entity[text]
        return en_entities

    place_name = None
    results = gmaps.places(text)
    if 'results' in results and results['results']:
        place_name = results['results'][0]['name']
    if place_name is None:
        geocode_result = gmaps.geocode(text)
        if geocode_result:
            if geocode_result[0]['address_components']:
                place_name = geocode_result[0]['address_components'][0]['long_name']
    mention2gmap_entity.cll.insert_one({"key": text, "value": place_name})

    return place_name


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

ORIYA_BASE = int('0x0B00', base=16)
ORIYA_MAX = int('0x0B7F', base=16)
DEVANAGARI_BASE = int('0x0900', base=16)
DEVANAGARI_MAX = int('0x097F', base=16)


def or2hindi(surface):
    hindi_str = ""
    for char in surface:
        tmp = ord(char) - (ORIYA_BASE-DEVANAGARI_BASE)
        if DEVANAGARI_BASE <= tmp <= DEVANAGARI_MAX:
            hindi_str += chr(tmp)
        else:
            hindi_str += char
    return hindi_str

