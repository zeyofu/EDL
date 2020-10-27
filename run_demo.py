import cherrypy
import json
import requests
from time import sleep
import link_entity_for_demo
import urllib
import pycountry

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
        json_ner = cherrypy.request.json
        lang = lang_converter(json_ner["lang"])

        edl_input = None
        for aview in json_ner["views"]:
            if aview["viewName"] == "NER_CONLL":
                edl_input = aview["viewName"]
                break
        return link_entity_for_demo.call_this_for_demo(edl_input, lang, 1)

if __name__ == '__main__':
    #
    # print("")
    # # INITIALIZE YOUR MODEL HERE IN ORDER TO KEEP IT IN MEMORY
    #
    # print("Starting rest service...")
    # config = {'server.socket_host': '0.0.0.0'}
    # cherrypy.config.update(config)
    # cherrypy.config.update({'server.socket_port': 8081})
    # cherrypy.quickstart(MyWebService())

    # # The target URL where we send to request to
    # url = 'http://dickens.seas.upenn.edu:4033/anns'
    # # The parameters we wish to send
    # json_in = {
    #     'text': 'Barack Obama is an American politician and attorney who served as the 44th president of the United States from 2009 to 2017.'}
    # # Set the headers for the request
    # headers = {'content-type': 'application/json'}
    #
    # json_out = None
    # while json_out is None:
    #     try:
    #
    #         # Post the request
    #         json_out = requests.post(url, data=json.dumps(json_in), headers=headers)
    #     except:
    #         sleep(5)
    #         print("Was a nice sleep, now let me continue...")
    #         continue

    # # Retrieve the JSON of the Output
    # res = json_out.json()

    url = 'https://cogcomp.seas.upenn.edu/dc4033/ner?lang=eng&model=bert&text=Barack%20Hussein%20Obama,%20an%20American%20politician%20serving%20as%20the%2044th%20President%20of%20the%20United%20States'
    res = urllib.request.urlopen(url).read().decode("utf-8")
    res_json = json.loads(res)
    # Print the response text (the content of the requested file):
    print(res_json)
    lang = 'aka'
    lang_code = lang_converter(lang)
    print(lang_code)