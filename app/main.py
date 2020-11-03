# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:10:28 2020
@author: tavakohr
"""
import spacy
from spacy.matcher import Matcher
from word2number import w2n
from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import os

path = os.getcwd()

nlp_reloaded = spacy.load(f'./updated-en-model')
nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.vocab.morphology.lemmatizer

LAMA_list = ['lama', 'long acting anticholinergic', 'long acting muscarinic agent', 'long acting muscarinic antagonist',
             'long-acting muscarinic antagonist', 'tiotropium', 'aclidinium', 'glycopyrronium'
    , 'umeclidinium', 'genuair', 'breezhaler', 'handiHaler', 'ellipta', 'respimat']

ICS_list = ['ics', 'inhaled corticosteroid', 'icslaba', 'symbicort', 'advair', 'dulera', 'fluticasone', 'flovent',
            'budesonide', 'pulmicort'
    , 'mometasone', 'asmanex', 'beclomethasone', 'qvar', 'ciclesonide', 'alvesco']

LABA_list = ['laba', 'long-acting beta-agonist', 'long-acting beta2-agonist', 'long acting beta2 agonist',
             'long acting beta 2 agonist', 'long-acting beta2-adrenoceptor', 'long acting beta agonist',
             'icslaba', 'symbicort', 'symbicort', 'advair', 'dulera', 'salmeterol', 'serevent', 'ormoterol',
             'perforomist']

STATIN_list = ['statin', 'atorvastatin', 'fluvastatin ', 'lovastatin', 'pravastatin', 'rosuvastatin', 'simvastatin',
               'lipitor', 'lescol',
               'pravachol', 'crestor', 'zocor']

Oxygen_list = ['oxygen therapy', 'o2', 'oxygen', 'o2 therapy']

ExSmoker_list = ['ex smoker', 'ex-smoker', 'former smoker']


#########################################  1-  Convert all word numbers  to digits

def text2int(textnum, numwords={}):
    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring


#########################################  2-  This is the internal function for binary finder ( negation)
def Negative_finder(text, itemlist):
    NER_negatives = []
    for item in itemlist:
        NER_negatives.append(item + ' ' + 'no')
        NER_negatives.append('no' + ' ' + item)

    neg_result = -1
    for neg_item in NER_negatives:
        if text.find(neg_item) > -1:
            neg_result = 0

    return (neg_result)


#########################################  3-  This is the internal function for binary finder
def positive_finder(text, itemlist):
    NER_positives = []
    for item in itemlist:
        NER_positives.append(item + ' ' + 'yes')
        NER_positives.append('yes' + ' ' + item)

    pos_result = -1
    for pos_item in NER_positives:
        if text.find(pos_item) > -1:
            pos_result = 1

    return (pos_result)


#########################################  4-  in case user forgot to say yes no , but mentioned entity it is assumed that it is positive
def nutral_finder(text, itemlist):
    NER_nutr = []
    for item in itemlist:
        NER_nutr.append(item)

    nutr_result = -1
    for nutr_item in NER_nutr:
        if text.find(nutr_item) > -1:
            nutr_result = 1

    return (nutr_result)


#########################################  5-  This is  the binary finder
def binary_finder(text, itemlist):
    try:
        p = positive_finder(text, itemlist)
        n = Negative_finder(text, itemlist)
        nut = nutral_finder(text, itemlist)
        return_v = -1
        if p == -1 and n == -1 and nut == -1:
            return_v = -1
        elif p == -1 and n == -1 and nut == 1:
            return_v = 1
        elif p != -1 and n == -1:
            return_v = p
        elif n != -1 and p == -1:
            return_v = n
        elif n != -1 and p != -1:
            return_v = n

        return ([str(return_v), 1])

    except Exception as e:
        print(e)


#########################################  6-  extracts numerical attribute of entities


def Relation_extractor_num(text, entity):
    try:

        number = -1
        pattern = [{'LOWER': entity.lower()},
                   {'LEMMA': 'be', 'OP': '?'},
                   {'IS_DIGIT': True}]

        matcher = Matcher(nlp_reloaded.vocab)
        matcher.add('EntityFinder', None, pattern)
        doc_m = nlp_reloaded(text.lower())
        matches = matcher(doc_m)
        # matches
        for matchid, start, end in matches:
            print(matchid)

            span = doc_m[start:end]
            small_doc = nlp_reloaded(span.text)
            for token in small_doc:

                if token.pos_ == 'NUM':
                    number = w2n.word_to_num(token.text)
                else:
                    pass

        return ([str(number), 1])

    except Exception as e:
        print(e)

    #########################################  7-  extracts the gender


def Gender_recognizer(doc_reloaded):
    ind = 0
    denom = 0

    try:
        for token in doc_reloaded:
            # print(token['Text']," " ,token['PartOfSpeech']['Tag'])
            if token.text.upper() == 'FEMALE':
                ind += 1
                denom += 1
            elif token.text.upper() == 'MALE':
                ind += -1
                denom += 1
            elif token.text.upper() == 'MAN':
                ind += -1
                denom += 1
            elif token.text.upper() == 'WOMAN':
                ind += 1
                denom += 1
            if token.text.upper() == 'SHE' and token.pos_ == 'PRON':
                ind += 1
                denom += 1
            elif token.text.upper() == 'HE' and token.pos_ == 'PRON':
                ind += -1
                denom += 1
            elif token.text.upper() == 'HIS' and token.pos_ == 'DET':
                ind += -1
                denom += 1
            elif token.text.upper() == 'HER' and token.pos_ == 'DET':
                ind += 1
                denom += 1

        if denom == 0:
            score = 0.5
        else:
            score = abs(ind / denom)

        # print(score)
        if score == 0.5:
            male = '-1'

        elif ind > 0:
            male = '0'
        else:
            male = '1'

        Out_Gender = male
        return ([Out_Gender, 1])
    except Exception as e:
        print(e)
        return (['-1', 0])
        print('Warning! incorrect input in Gender_recognizer')

    #########################################  8-  extracts the smoking status


def Smoking_recognizer(doc_reloaded):
    doc = doc_reloaded
    sents = [sent for sent in doc.sents]
    smoker = -1

    try:
        out_loop = '-1'  # default valu is -1 which means function couldn't find the smocking entity
        for sent in sents:
            negation = 0

            negation = [tok for tok in sent if tok.dep_ == 'neg']

            quitted = [tok for tok in sent if tok.lemma_ == 'quit' or tok.lemma_ == 'no' or tok.lemma_ == 'not'
                       or tok.lemma_ == 'non' or tok.lemma_ == 'ex']

            for it in ExSmoker_list:
                if sent.text.find(it) > -1:
                    smoker = 0

            # print(sent, any(tok for tok in sent if tok.lemma_ == 'smoke' or tok.lemma_ == 'smoker'or tok.lemma_ == 'smoking'))
            if any(tok for tok in sent if tok.lemma_ == 'smoke' or tok.lemma_ == 'smoker' or tok.lemma_ == 'smoking'):

                if len(negation) > 0 or len(quitted) > 0 or smoker == 0:
                    out_loop = '0'
                else:
                    out_loop = '1'
        if out_loop != '-1':
            out = out_loop

        else:
            out = '-1'

        # score=score_calculator(doc_reloaded,"smoke")
        out_smoking = out

        return ([out_smoking, 1])

    except Exception as e:
        print(e)


#########################################  9-  Lambda functoin to export all the results into json


def lambda_handler(text, event=None, context=None):
    originaltext = text
    text = text.lower()

    text = text.replace(',', ' , ')
    text = text.replace('\n', ' , ')
    text = text.replace('next', ' , ')

    text = text2int(text)
    text = text.replace('zero', '0')

    pre_doc = nlp_reloaded(text)

    for token in pre_doc:

        if token.tag_ == 'NNS':
            singular = lemmatizer(token.text, NOUN)[0]
            print(token.text, token.tag_, singular)

            # print(singular,token.text)
            text = text.replace(token.text, singular)

    doc_reloaded = nlp_reloaded(text)

    try:
        #############  Part 1: Analyse text with 4 NLP Models

        if len(text) >= 5000:
            print('text is either more than 5000 bytes or empty')
            return
        elif text != '':

            ###############  Part 2: Create output dictionary
            global_dict = {}
            global_dict['male'] = Gender_recognizer(doc_reloaded)
            global_dict['Age'] = Relation_extractor_num(text, 'age')
            global_dict['smoker'] = Smoking_recognizer(doc_reloaded)
            global_dict['FEV1'] = Relation_extractor_num(text, 'fev1')
            global_dict['SGRQ'] = Relation_extractor_num(text, 'SGRQ')
            global_dict['CAT'] = Relation_extractor_num(text, 'cat')
            global_dict['BMI'] = Relation_extractor_num(text, 'bmi')
            global_dict['oxygen'] = binary_finder(text, Oxygen_list)
            global_dict['Statin'] = binary_finder(text, STATIN_list)
            global_dict['LAMA'] = binary_finder(text, LAMA_list)
            global_dict['LABA'] = binary_finder(text, LABA_list)
            global_dict['ICS'] = binary_finder(text, ICS_list)
            global_dict['LastYrExacCount'] = Relation_extractor_num(text, 'exacerbation')
            global_dict['LastYrSevExacCount'] = Relation_extractor_num(text, 'hospitalization')
            global_dict['text'] = originaltext

            return {'statusCode': 200, 'body': global_dict}


    except Exception as e:
        print(e)


text = '''
Age 87,
sex Female,
BMI 26,
Smoking non smoker,
Oxygen  No ,
LAMA no,
yes LABA ,
ICS no,
Statin no,,
FEV1 41
Number of exacerbations one
Hospitalizations  one,
SGRQ 20
'''

############################### app.py - a minimal flask api using flask_restful
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


@app.route('/api', methods=['POST'])
def nlpaccept():
    analyzed = lambda_handler(request.json['dictation'])
    return jsonify(analyzed)


@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        text = request.form['text']
        out = lambda_handler(text)
        return render_template('text.html', text=out)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')