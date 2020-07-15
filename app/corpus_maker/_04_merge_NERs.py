# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 18:43:52 2020

@author: tavakohr
"""


import spacy # tested with v2.2.3
from spacy.pipeline import EntityRecognizer
import os
os.chdir(r'C:\Users\tavakohr\Documents\Accept\env\Accept_NLP')
path=os.getcwd()
 

text = ''' She uses LAMA  and long acting anticholinergics and LABA and ICS'''             

# load the English and My COPD  moedls
nlp_en = spacy.load('en_core_web_sm')  # NER tags PERSON, GPE, ...


doc=nlp_en(text)
[(ent, ent.label_) for ent in doc.ents]

nlp_ex = spacy.load(f'{path}\\corpus_maker\\Model') # NER tags PER, LOC, ...
 
[(ent,ent.label_) for ent in nlp_ex(text).ents]

# the Vocab objects are not the same
assert nlp_en.vocab != nlp_ex.vocab

# but the vectors are iexntical (because neither moedl has vectors)
assert nlp_en.vocab.vectors.to_bytes() == nlp_ex.vocab.vectors.to_bytes()

# original English output



# original COPD output  
doc2 = nlp_ex(text)
print([(ent.text, ent.label_) for ent in doc2.ents])


# initialize a new NER component with the vocab from the English pipeline
ner_ex = EntityRecognizer(nlp_en.vocab)

# reload the NER component from the COPD moedl by serializing
# without the vocab and exserializing using the new NER component
ner_ex.from_bytes(nlp_ex.get_pipe("ner").to_bytes(excluex=["vocab"]))

# add the COPD NER component to the end of the English pipeline
nlp_en.add_pipe(ner_ex, name="ner_ex")

# check that they have the same vocab
assert nlp_en.vocab == ner_ex.vocab

# combined output (English NER runs first, COPD second)
doc3 = nlp_en(text)
print([(ent.text, ent.label_) for ent in doc3.ents])
# [('Jane', 'PERSON')
nlp_en.to_disk(f'{path}\\updated-en-model')






nlp_reloaded = spacy.load(f'{path}\\updated-en-model',disable = ['ner'])
print(nlp_reloaded.pipe_names)

   
text=   '''what is the price of   
inhaled corticosteroids 
inhaled corticosteroid 
ICS 
long-acting beta2-agonist 
long acting beta2 agonist 
LABA 
long-acting beta2-adrenoceptor 
long-acting beta-agonist 
long acting muscarinic antagonist 
long acting anticholinergic  
LAMA 
long-acting muscarinic antagonist 
severe exacerbation 
severe COPD exacerbation 
STATIN 
long-acting beta2-agonists 
long acting beta2 agonists 
LABAs 
long-acting beta2-adrenoceptors 
long-acting beta-agonists 
long acting muscarinic antagonists 
long acting anticholinergics 
LAMAs 
long-acting muscarinic antagonists 
severe exacerbations 
severe COPD exacerbations 
O2 therapy
Oxygen therapy 
never used statin '''  
    
doc=nlp_reloaded(text)   
    
for ent in doc.ents:
    print(f"{str(ent.text):{30}} ,{str(ent.label_):{20}}"  )
    


    




