import glob
import itertools
import re
# from write_data_to_annotate import camel
from ccg_nlpy import TextAnnotation
import logging


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def camel(s):
    return s != s.lower() and s != s.upper()


def correct_surface(q, lang):
    corrects = []
    if lang == 'tl':
        if 'tikong' in q:
            corrects.append(q.replace('tikong', 'te'))
    if lang == 'ur':
        if q[-1] == 'ی':
            corrects.append(q[:-1])
    if lang == "ilo":
        if "Philippines" in q:
            q_processed = q.replace('Philippines', 'Philippine')
            corrects.append(q_processed)
        if "Philippine" in q:
            q_processed = q.replace('Philippine', 'Philippines')
            corrects.append(q_processed)

    if lang == "or":
        if '଼' in q:
            q_processed = q.replace('଼', '')
            corrects.append(q_processed)
        if "ଡ" in q:
            q_processed = q.replace("ଡ", "ଡ଼")
            corrects.append(q_processed)

        if "ଡ଼" in q:
            q_processed = q.replace("ଡ଼", "ଡ")
            corrects.append(q_processed)

        if "ଡ" in q:
            q_processed = q.replace("ଡ", "ଡ଼")
            corrects.append(q_processed)

        if "ଡ଼" in q:
            q_processed = q.replace("ଡ଼", "ଡ")
            corrects.append(q_processed)

        if "ବ" in q:
            q_processed = q.replace("ବ", "ଵ")
            corrects.append(q_processed)

        if "ଵ" in q:
            q_processed = q.replace("ଵ", "ବ")
            corrects.append(q_processed)

        if 'ା' in q:
            q_processed = q.replace("ଵ", "ବ")
            corrects.append(q_processed)

        if "ଢ଼" in q:
            q_processed = q.replace("ଢ଼", "ଢ")
            corrects.append(q_processed)


        if "ଢ" in q:
            q_processed = q.replace("ଢ", "ଢ଼")
            corrects.append(q_processed)

        if 'ଲ୍ଳ' in q:
            q_processed = q.replace('ଲ୍ଳ', 'ଲ୍ଲ')
            corrects.append(q_processed)

        if 'ଲ୍ଲ' in q:
            q_processed = q.replace('ଲ୍ଲ','ଲ୍ଳ')
            corrects.append(q_processed)

    corrects += remove_prefix(q, lang)
    return corrects


def remove_prefix(s, lang):
    ans = []
    prefixes = {
                "rw":
                    ["z’", "y’", "w’", "ry’", "n’", "rw’", "m’", "c’", "cy’", "b’", "by’", "bw’"]
                }
    if lang in prefixes:
        for pre in prefixes[lang]:
            if s.startswith(pre):
                # logging.info("removing known prefix %s", pre)
                ans.append(s[len(pre):])
    return ans


def remove_suffix(s, lang):
    # make a list of known suffixes and abstract this
    suffixes = {"si":
                    ["නු", "න", "ට", "දී"],
                "rw":
                    [],
                "or":
                    ['ଠାରେ', 'ସ୍ଥିତ', 'ରେ', 'କୁ', 'ର', 'ଙ୍କ', 'ୀ', "ରୁ", "ଲାରୁ","ଙ୍କ", "୯.୮",'୍'],
                "ilo":
                    [],
                "ti":["ஆ", "ச", "ப", "ம", "ல", "வ"]
                }
    if lang in suffixes:
        for suff in suffixes[lang]:
            # print(list(s),list(suff))
            if s.endswith(suff):
                logging.info("removing known suffix %s", suff)
                s = s[:-len(suff)]
                return s
    return s


def remove_suffix_2(s, k):
  return s[:-k]


def add_suffix(s, lang):
    ans = []
    # make a list of known suffixes and abstract this
    suffixes = {"si":
                    [ "ව", "යා","ය", "ා", "ට", "ෙන්", "ෙහි"]
                }
    if lang in suffixes:
        for suff in suffixes[lang]:
            ans.append(s + suff)
    return ans


def replace_suffix(s, lang):
    ans = []
    replace = {"si":
                    ["වේ", "නු"]
                }
    if len(s) > 2 and lang in replace:
        for r in replace[lang]:
            ans.append(s[-1] + r)
    return ans


def remove_consecutive_duplicates(word):
    ans = []
    removed = ''.join(i for i, _ in itertools.groupby(word))
    if removed != word:
        ans.append(removed)
    return ans


def load_ta_from_jsons(json_dir):
    doc2ta = {}
    tafiles = glob.glob(json_dir + "/*")
    for tafile in tafiles:
        ta = TextAnnotation(json_str=open(tafile).read())
        docid = tafile.split("/")[-1]
        doc2ta[docid] = ta
    return doc2ta
