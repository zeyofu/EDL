import cherrypy
import json
import requests
from time import sleep
import link_entity_for_demo
import urllib
import pycountry
from ccg_nlpy.core.text_annotation import TextAnnotation

def lang_converter(lang):
    if len(lang) == 3 and lang != 'ilo':
        language = pycountry.languages.get(alpha_3=lang)
        if (language != None):
            lang = (language.alpha_2)
        else:
            print("the language code is not covered")
    return lang


class MyWebService(object):

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def edl(self):
        # input_json_ner_data
        json_input = cherrypy.request.json
        lang = json_input["lang"]  # NER language (2/3 chars)
        text = json_input["text"]
        url = f'https://cogcomp.seas.upenn.edu/dc4033/ner?lang={lang}&model=bert&text={text}'
        res = urllib.request.urlopen(url).read().decode("utf-8")
        edl_input = json.loads(res)
        lang = lang_converter(lang)

        return link_entity_for_demo.call_this_for_demo(edl_input, lang, 1)

if __name__ == '__main__':

    print("")
    # INITIALIZE YOUR MODEL HERE IN ORDER TO KEEP IT IN MEMORY

    print("Starting rest service...")
    config = {'server.socket_host': '0.0.0.0'}
    cherrypy.config.update(config)
    cherrypy.config.update({'server.socket_port': 8081})
    cherrypy.quickstart(MyWebService())

    # # Retrieve the JSON of the Output
    # res = json_out.json()

    # url = 'https://cogcomp.seas.upenn.edu/dc4033/ner?lang=eng&model=bert&text=Barack%20Hussein%20Obama,%20an%20American%20politician%20serving%20as%20the%2044th%20President%20of%20the%20United%20States'
    # res = urllib.request.urlopen(url).read().decode("utf-8")
    # res_json = json.loads(res)
    # # Print the response text (the content of the requested file):
    # print(res_json)
    # lang = 'aka'
    # lang_code = lang_converter(lang)
    # print(lang_code)
    #
    # # example input file: /pool0/webserver/incoming/experiment_tmp/EDL2019/data/input/ak/AKA_NA_006644_20170516_H0025ZXL0
    # docta = TextAnnotation(json_str=open('/pool0/webserver/incoming/experiment_tmp/EDL2019/data/input/example.txt', encoding='utf8', errors='ignore').read())
    # link_entity_for_demo.call_this_for_demo(docta, 'so', 1)
