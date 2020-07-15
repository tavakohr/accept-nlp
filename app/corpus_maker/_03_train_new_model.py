# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:03:10 2020

@author: tavakohr
"""


import spacy
import random
import os
os.chdir(r'C:\Users\tavakohr\Documents\Accept\env\Accept_NLP')

 
path=os.getcwd()
 
################################################################################################ import the training set and create the function
f = open(f'{path}\\corpus_maker\\_22_tupels.txt','r',encoding="utf8")
Large_corpus_imported =[]
for line in f:
    if line[0]=="(":
        tup = eval(line)
        Large_corpus_imported.append(tup)
 

TRAIN_DATA =  Large_corpus_imported





def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp





prdnlp = train_spacy(TRAIN_DATA, 100)


prdnlp.to_disk(f'{path}\\corpus_maker\\Model')
new_nlp=spacy.load(f'{path}\\corpus_maker\\Model') 
#Test your text


test_text =  '''what is the price of   
1-she doesnt use inhaled corticosteroids 
2-inhaled corticosteroid 
3-ICS 
4 long-acting beta2-agonist 
5 long acting beta2 agonist 
6 LABA 
7 long-acting beta2-adrenoceptor 
8 long-acting beta-agonist 
9 long acting muscarinic antagonist 
10 long acting anticholinergic  
11 LAMA 
12 long-acting muscarinic antagonist 
13 severe exacerbation 
14 severe COPD exacerbation 
15 STATIN 
16 long-acting beta2-agonists 
17 long acting beta2 agonists 
18 LABAs 
19 long-acting beta2-adrenoceptors 
20 long-acting beta-agonists 
21 long acting muscarinic antagonists 
22 long acting anticholinergics 
23 LAMAs 
24 long-acting muscarinic antagonists 
25 severe exacerbations 
26 severe COPD exacerbations 
27 STATINs 
long acting beta agonist 
Oxygen therapy
O2 Therapy.
'''

test_text =  '''
LAMA
long acting anticholinergics 
long acting muscarinics 
LAMA
long acting anticholin 
long acting muscar 
'''


doc = new_nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
   