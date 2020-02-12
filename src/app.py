from flask import request, jsonify, make_response
from MASS import Translator
from src import app

import os


en2zh_translator = Translator()
en2zh_translator.initialize()
zh2en_translator = Translator()
zh2en_translator.initialize(s_lang='zh', t_lang='en',
                            bpe_codes=os.path.join(os.path.dirname(__file__), '../MASS/model/bpe/all.zh.bpe.codes'))


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/translate', methods=['POST'])
def translate_sentence():
    params = request.json
    s_lang = params.get('s_lang')
    s_text = params.get('s_text')

    if s_lang == 'en':
        t_text = en2zh_translator.translate(s_text)
    elif s_lang == 'zh':
        t_text = zh2en_translator.translate(s_text)
    else:
        t_text = 'Wrong source language!'
    json_obj = {'s_text': s_text,
                't_text': t_text}
    response = make_response(jsonify(json_obj), 200)
    return response
