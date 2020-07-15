# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 17:56:51 2020

@author: tavakohr


"""

def heyhey ():
    print ("hey hey")


###########################################################################################  Part 0 :import libraries
# import json
import spacy
from spacy.matcher import Matcher
from collections import defaultdict
import boto3
import os

# import requests

AWS_M_client  = boto3.client(service_name='comprehendmedical', region_name='ca-central-1',
                      aws_access_key_id='AKIA5NP4GPLTCRHVHBHD',
                      aws_secret_access_key='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')
#AWS_C_client = boto3.client(service_name='comprehend', region_name='ca-central-1',
#                        aws_access_key_id='AKIA5NP4GPLTCRHVHBHD',
#                        aws_secret_access_key='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')
path = os.getcwd()

nlp_reloaded = spacy.load(f'./updated-en-model')

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

text = '''
patient is a 78 years old female, she is not a smoker. 
had 5 severe exacerbation last year.
She had 2 moderate exacerbations last year.
she is under treatment with budesonide and umeclidinium . 
she never used any statins last year. 
CAT score is 23 and her BMI is 31.
she do not use  LABA. 
she hasn't used LABA. Never used LABA . 
She doesn't use long acting anticholinergic.
FEV1 percentage is 25%.
Dr J sent his suggestions for treatment,
she is not on Oxygen therapy.
SGRQ was 23'''
doc_reloaded = nlp_reloaded(text)


############################################################################################ part 2: general functions

# tests if a key exist in a dictionaly or not------------------------------------------------


def checkKey(dict, key):
    if key in dict.keys():
        return 1
    else:
        return 0
    #


# converts the test numbers to the digital numerics------------------------------------------

def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

    # caclulates the probability of each entity within the test----------------------------------


def score_calculator(text, entity):
    try:

        # with nlp_reloaded2.disable_pipes('ner'):
        out = 0
        doc_reloaded = nlp_reloaded(text)
        if entity in [ent.label_ for ent in doc_reloaded.ents]:
            # print(entity,[ent.label_ for ent in doc_reloaded.ents])

            threshold = 0.2
            beams = nlp_reloaded.entity.beam_parse([doc_reloaded], beam_width=16, beam_density=0.0001)

            entity_scores = defaultdict(float)
            for beam in beams:

                for score, ents in nlp_reloaded.entity.moves.get_beam_parses(beam):

                    for start, end, label in ents:
                        # print(start, end, label)
                        entity_scores[(start, end, label)] += score

            # print ('Entities and scores (detected with beam search)')
            entitySecors = {}
            for key in entity_scores:
                start, end, label = key
                score = entity_scores[key]
                if (score > threshold):
                    entitySecors[label] = score

            out = entitySecors[entity]

        else:
            out = 0



    except TypeError:
        print('Warning! incorrect input in score_calculator', NameError)
        return (0)

    return (round(out, 2))


# score_calculator(text ,'LABA')


entity = 'OXYGEN'


###########################################################################################  Part 3: Boolean variables

def Boolean_recognizer(text, doc_reloaded, entity, AWS_M_entities, array):
    arr = array
    negation = 0
    out = '0'
    out_aws = '0'
    score = 0
    score_list = []
    is_entity_AWS = 0
    try:

        # 1- AWS section , look at AWS entities
        for ent in AWS_M_entities:
            if any(c in str(ent['Text']).lower() for c in arr):
                score = ent['Score']
                score_list.append(score)
                is_entity_AWS = 1

                if len(ent['Traits']) > 0:
                    if checkKey(ent['Traits'][0], 'Name') == 1:
                        negation = 1

                    else:
                        negation = 0

                if is_entity_AWS == 1 and negation == 1:
                    out_aws = '-1'

                elif is_entity_AWS == 1 and negation == 0:
                    out_aws = '1'
                    # print(is_entity_AWS ,ent)

        # 2-1 SpaCy-entity section, look at doc_reloaded entities

        for ent in doc_reloaded.ents:
            # print(ent)
            if str(ent).lower() in arr:
                score_list.append(1)  # score_calculator(text.lower(),entity))

        sents = [sent for sent in doc_reloaded.sents]

        out_loop = '0'
        for sent in sents:
            negation = '0'

            is_entity = entity.lower() in [ent.label_.lower() for ent in sent.ents] or entity.lower() in [
                ent.text.lower() for ent in sent.ents]

            is_negation = 'neg' in [tok.dep_ for tok in sent if tok.dep_ == 'neg'] or len(
                [tok for tok in sent if tok.text.lower() == 'no']) > 0
            # print(is_negation ,is_entity)

            if is_negation == True and is_entity == True:
                out_loop = '-1'


            elif is_negation == False and is_entity == True:
                out_loop = '1'

            if out_loop != '0':
                out = out_loop
            else:
                out = '0'

        # if AWS finds anything but my code didn't , it means I didn't trained for that specific entity, so the AWS results is final
        # but if I found -1 and AWS return anything . my code override the AWS
        # otherwise AWS

        if out_aws == '1' and out == '0':
            final_output = '1'
        elif out == '-1':
            final_output = '-1'
        else:
            final_output = out_aws

        Out_return = [final_output, round(max(score_list), 3)]
        return (Out_return)

    except Exception:
        print(f'Warning! incorrect input in Boolean_recognizer {entity}')
        return (['0', 0])


# Boolean_recognizer(text,doc_reloaded,'STATIN',AWS_M_entities,STATIN_list)
# Boolean_recognizer(text,doc_reloaded,'OXYGEN',AWS_M_entities,Oxygen_list)
# Boolean_recognizer(text,doc_reloaded,'ICS',AWS_M_entities,ICS_list)

################################################################################################   Part 4: Gender


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

        if ind > 0:
            Gender = "Female"
        else:
            Gender = "male"

        Out_Gender = [Gender, round(score, 3)]
        return (Out_Gender)
    except Exception:
        print('Warning! incorrect input in Gender_recognizer')
        return ('0')


################################################################################################   Part 5.1: Smoking history
def Smoking_recognizer(doc_reloaded):
    doc = doc_reloaded
    sents = [sent for sent in doc.sents]

    try:
        out_loop = '0'
        for sent in sents:
            negation = 0

            negation = [tok for tok in sent if tok.dep_ == 'neg']

            quitted = [tok for tok in sent if tok.lemma_ == 'quit' or tok.lemma_ == 'no' or tok.lemma_ == 'not']
            # print(sent, any(tok for tok in sent if tok.lemma_ == 'smoke' or tok.lemma_ == 'smoker'or tok.lemma_ == 'smoking'))
            if any(tok for tok in sent if tok.lemma_ == 'smoke' or tok.lemma_ == 'smoker' or tok.lemma_ == 'smoking'):

                if len(negation) > 0 or len(quitted) > 0:
                    out_loop = '-1'
                else:
                    out_loop = '1'
        if out_loop != '0':
            out = out_loop
            score = 1
        else:
            out = '0'
            score = 0

        # score=score_calculator(doc_reloaded,"smoke")
        out_smoking = [out, round(score, 3)]

        return (out_smoking)
    except Exception:
        print('Warning! incorrect input in Smoking_recognizer')
        return (['0', 0])


# Smoking_recognizer(doc_reloaded)

################################################################################################   Part 6: Severe exacerbations

def Number_Severe_exacerbation(text, doc_reloaded):
    n = '0'
    sent_n = '0'
    score = 0
    sents = [sent for sent in doc_reloaded.sents]
    # sent=sents[2]

    try:

        for sent in sents:

            severe_token_exacerbation = 0
            sent_entity_exacerbation = 0
            N_exacerbation = 0

            negation_list = []
            neg_words_in_sentence = [tok for tok in sent if tok.text.lower() == 'no']
            neg_token_dependency = [tok for tok in sent if tok.dep_ == 'neg']
            if len(neg_words_in_sentence) > 0:
                negation_list.append(neg_words_in_sentence)

            if len(neg_token_dependency) > 0:
                negation_list.append(neg_token_dependency)

            negation = len(negation_list)
            # print(negation)

            # following I used both token method and entity emethod to identify severe exacerbation if eiother happentd I count as severe exacerbation
            for ent in sent.ents:

                if str(ent.label_) == 'SevereExacerbation':
                    score = 1  # score_calculator(text,'SevereExacerbation')
                    # print (ent.label_)

                    sent_entity_exacerbation = 1

            for token in sent:
                print(token, token.dep_)
                for child in token.children:
                    # print(f'{token } , {token.dep_ } ----->{child},{child.dep_}')

                    # if (token.text.lower()=='exacerbation' or token.text.lower()=='exacerbations'  ):
                    # print(f" {str(token.text):<15}   {token.dep_:<10}  {child}","----->" ,child.dep_)

                    if (
                            token.text.lower() == 'exacerbation' or token.text.lower() == 'exacerbations') and child.dep_ == 'nummod':
                        N_exacerbation = child
                        # print(child)
                    if (
                            token.text.lower() == 'exacerbation' or token.text.lower() == 'exacerbations') and child.text.lower() == 'severe':
                        severe_token_exacerbation = 1
            # print(severe_token_exacerbation, N_exacerbation)

            if severe_token_exacerbation == 1:
                if w2n.word_to_num(str(N_exacerbation)) > w2n.word_to_num(str(sent_n)):
                    sent_n = str(w2n.word_to_num(str(N_exacerbation)))
                    # print(sent_n)

            if (severe_token_exacerbation == 1 or sent_entity_exacerbation == 1) and negation == 1:
                n = '-1'
            elif (severe_token_exacerbation == 1 or sent_entity_exacerbation == 1) and negation == 0:
                n = sent_n

        out_ = [n, round(score, 3)]
        return (out_)
    except Exception:
        print('Warning! incorrect input , Number_Severe_exacerbation function')
        return (['0', 0])


# Number_Severe_exacerbation(text,doc_reloaded)


################################################################################################   Part 7: Mild exacerbations


def Number_Mild_exacerbation(doc_reloaded):
    n = '0'
    sent_n = '0'

    sents = [sent for sent in doc_reloaded.sents]

    sent = sents[1]
    try:

        for sent in sents:
            if 'exacerbation' in sent.text:

                severe_token_exacerbation = 0
                N_exacerbation = 0

                negation_list = []
                neg_words_in_sentence = [tok for tok in sent if tok.text.lower() == 'no']
                neg_token_dependency = [tok for tok in sent if tok.dep_ == 'neg']
                if len(neg_words_in_sentence) > 0:
                    negation_list.append(neg_words_in_sentence)

                if len(neg_token_dependency) > 0:
                    negation_list.append(neg_token_dependency)

                negation = len(negation_list)
                # print(negation)

                for token in sent:
                    # print(token,token.dep_)
                    for child in token.children:

                        # if (token.text.lower()=='exacerbation' or token.text.lower()=='exacerbations'  ):
                        # print(f" {str(token.text):<15}   {token.dep_:<10}  {child}","----->" ,child.dep_)

                        if (
                                token.text.lower() == 'exacerbation' or token.text.lower() == 'exacerbations') and child.dep_ == 'nummod':
                            N_exacerbation = child
                            # print(child)
                        if (
                                token.text.lower() == 'exacerbation' or token.text.lower() == 'exacerbations') and child.text.lower() == 'severe':
                            severe_token_exacerbation = 1

                # print(severe_token_exacerbation, N_exacerbation)

                if severe_token_exacerbation != 1:
                    if w2n.word_to_num(str(N_exacerbation)) > w2n.word_to_num(str(sent_n)):
                        sent_n = str(w2n.word_to_num(str(N_exacerbation)))
                        print(sent_n)

                if (severe_token_exacerbation == 0) and negation == 1:
                    n = '-1'
                    print(sent)
                elif (severe_token_exacerbation == 0) and negation == 0:
                    n = sent_n

            # score=score_calculator(text,'SevereExacerbation')
            out_ = [n, 1.0]
        return (out_)
    except Exception:
        print('Warning! incorrect input, Number_Mild_exacerbation function')
        return (['0', 0])


# Number_Mild_exacerbation(doc_reloaded)

################################################################################################   Part 8: Continues Variables

################################################# Age ################################
def Age_recognizer(AWS_M_entities):
    score = 0
    AWS_M_entities = AWS_M_entities
    Age_loop = '0'
    try:
        for entity in AWS_M_entities:
            # print('Entity ', i, entity)
            if entity['Type'] == 'AGE':
                score = entity['Score']
                Age_loop = entity['Text']
        if Age_loop != '0':

            if is_number_tryexcept(Age_loop) == True:
                Age = str(Age_loop)
            else:
                Age = str(w2n.word_to_num(str(Age_loop)))

        else:
            Age = 0

        Age_return = [Age, round(score, 3)]
        return (Age_return)
    except Exception:
        print('Warning! incorrect input in Age_recognizer')
        return (['0', 0])
    # Age_recognizer(AWS_M_entities)


################################################# BMI ################################

def BMI_recognizer(AWS_M_entities):
    arr = ['bmi', 'body mass index']
    BMI_loop = '0'
    score = 0
    try:
        for entity in AWS_M_entities:

            string = str(entity['Text']).lower()
            if any(c in string for c in arr):
                score = entity['Score']
                # print(entity)

                if checkKey(entity, 'Attributes') == 1:

                    if checkKey(entity['Attributes'][0], 'Text') == 1:
                        BMI_loop = entity['Attributes'][0]['Text']
                        # print( entity['Attributes'][0]['Text'])

            if BMI_loop != '0':
                BMI = str(w2n.word_to_num(str(BMI_loop)))
            else:
                BMI = '0'

        BMI_return = [BMI, round(score, 3)]

        return (BMI_return)
    except Exception:
        print('Warning! incorrect input in BMI_recognizer ')
        return (['0', 0])


# BMI_recognizer(AWS_M_entities)


################################################# SGRQ ################################
def SGRQ_recognizer(AWS_M_entities):
    arr = ['respiratory questionnaire score', 'sgrq', 'st george score', "st. george's respiratory questionnaire"]
    SGRQ_loop = '0'
    score = 0
    try:
        for entity in AWS_M_entities:
            string = str(entity['Text']).lower()
            if any(c in string for c in arr):
                score = entity['Score']

                if checkKey(entity, 'Attributes') == 1:

                    # print( entity['Attributes'][0]['Text'] )
                    if checkKey(entity['Attributes'][0], 'Text') == 1:
                        SGRQ_loop = entity['Attributes'][0]['Text']

        if SGRQ_loop != '0':
            if is_number_tryexcept(SGRQ_loop) == True:
                SGRQ = str(SGRQ_loop)
            else:
                SGRQ = str(w2n.word_to_num(str(SGRQ_loop)))
        else:
            SGRQ = '0'

        out_ = [SGRQ, round(score, 3)]
        return (out_)
    except Exception:
        print('Warning! incorrect input in SGRQ_recognizer')
        return (['0', 0])


# SGRQ_recognizer(AWS_M_entities)

################################################# CAT ################################
def CAT_recognizer(AWS_M_entities):
    arr = ['copd assessment', 'cat score', 'cat']
    CAT_loop = '0'
    score = 0

    try:
        for entity in AWS_M_entities:
            string = str(entity['Text']).lower()
            if any(c in string for c in arr):
                score = entity['Score']

                if checkKey(entity, 'Attributes') == 1:

                    # print( entity['Attributes'][0]['Text'] )
                    if checkKey(entity['Attributes'][0], 'Text') == 1:
                        CAT_loop = entity['Attributes'][0]['Text']

        if CAT_loop != '0':
            if is_number_tryexcept(CAT_loop) == True:
                CAT = str(CAT_loop)
            else:
                CAT = str(w2n.word_to_num(str(CAT_loop)))

        else:
            CAT = '0'

        out_ = [CAT, round(score, 3)]

        return (out_)

    except Exception:
        print('Warning! incorrect input in CAT_recognizer')
        return (['0', 0])


# CAT_recognizer(AWS_M_entities)


#################################   FEV1 ###################################################

def FEV1_recognizer(AWS_M_entities):
    arr = ['fev1']
    FEV1_loop = '0'
    score = 0
    try:
        for entity in AWS_M_entities:
            string = str(entity['Text']).lower()
            if any(c in string for c in arr):
                score = entity['Score']

                if checkKey(entity, 'Attributes') == 1:

                    # print( entity['Attributes'][0]['Text'] )
                    if checkKey(entity['Attributes'][0], 'Text') == 1:
                        FEV1_loop = entity['Attributes'][0]['Text']

        if FEV1_loop != '0':
            FEV1 = FEV1_loop
        else:
            FEV1 = '0'

        if FEV1_loop != '0':
            if is_number_tryexcept(FEV1_loop) == True:
                FEV1 = str(FEV1_loop)
            else:
                FEV1 = str(w2n.word_to_num(str(FEV1_loop)))

        else:
            FEV1 = '0'

        out_ = [FEV1, round(score, 3)]

        return (out_)

        return (FEV1)
    except Exception:
        print('there is not a correct input in FEV1_recognizer')
        return (['0', 0])


# FEV1_recognizer(AWS_M_entities)

###########################################################################################  Part 1 :Add Text


def lambda_handler(text, event=None, contex=None):
    originaltext = text

    try:
        #############  Part 1: Analyse text with 4 NLP Models

        if len(text) >= 5000:
            print('text is either more than 5000 bytes or empty')
            return
        elif text != '':

            fulldocument = text.replace(" and ", ". and ")
            fulldocument = fulldocument.replace(",and ", ". and ")
            text = fulldocument

            AWS_M_result = AWS_M_client.detect_entities(Text=text)
            AWS_M_entities = AWS_M_result['Entities']
            # AWS_C_tokens=AWS_C_client.detect_syntax(Text=   text,LanguageCode='en')
            doc_reloaded = nlp_reloaded(text)

            ###############  Part 2: Create output dictionary
            global_dict = {}
            global_dict['Gender'] = Gender_recognizer(doc_reloaded)
            global_dict['Age'] = Age_recognizer(AWS_M_entities)
            global_dict['smoker'] = Smoking_recognizer(doc_reloaded)
            global_dict['FEV1'] = FEV1_recognizer(AWS_M_entities)
            global_dict['SGRQ'] = SGRQ_recognizer(AWS_M_entities)
            global_dict['CAT'] = CAT_recognizer(AWS_M_entities)
            global_dict['BMI'] = BMI_recognizer(AWS_M_entities)
            global_dict['Oxygen_Therapy'] = Boolean_recognizer(text, doc_reloaded, 'OXYGEN', AWS_M_entities,
                                                               Oxygen_list)
            global_dict['Statin'] = Boolean_recognizer(text, doc_reloaded, 'STATIN', AWS_M_entities, STATIN_list)
            global_dict['LAMA'] = Boolean_recognizer(text, doc_reloaded, 'LAMA', AWS_M_entities, LAMA_list)
            global_dict['LABA'] = Boolean_recognizer(text, doc_reloaded, 'LABA', AWS_M_entities, LABA_list)
            global_dict['ICS'] = Boolean_recognizer(text, doc_reloaded, 'ICS', AWS_M_entities, ICS_list)
            global_dict['Exacerbations'] = Number_Mild_exacerbation(doc_reloaded)
            global_dict['Severe_exacerbation'] = Number_Severe_exacerbation(text, doc_reloaded)
            global_dict['text'] = originaltext

            return {'stausCode': 200, 'body': global_dict}


    except:
        if len(text) < 5000:
            print('file size has an issue')
        else:
            print('Warning! incorrect input in lambda_handler')


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

