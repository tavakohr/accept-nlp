# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 21:37:19 2020

@author: tavakohr
"""

import os
import pandas as pd
import glob 
import spacy

 
#nlp=spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_sm', disable = ['ner', 'tagger', 'parser', 'textcat'])
nlp.add_pipe(nlp.create_pipe('sentencizer')) 


nlp.max_length = 103105000
os.chdir(r'C:\Users\tavakohr\Documents\Accept\env\Accept_NLP')
path=os.getcwd()

#import list of medications and their conterparts and classes from the csv file    
medication_list=pd.read_csv(f'{path}\\corpus_maker\\_11_entities_singular.csv')

list_of_names = medication_list['medname'].values.tolist()
entity_label = medication_list['class'].values.tolist()
entity_length = medication_list['length'].values.tolist()


#Extract all pubmed text's pathes from the folder
def filebrowser(ext=""):
    "Returns files with an extension"
    return [f for f in glob.glob(f"{path}\\corpus_maker\\PubMeds\\*.txt")]

filepathes = filebrowser(".txt")


# remove unnecessary breaks from pubmed abstracts
fulldocument='' 
for file in filepathes:        
    with open(file,encoding="utf8") as f:
        text = f.read()
        fulldocument+='.\n '+text    
        fulldocument = fulldocument.replace("\n","")
        fulldocument = fulldocument.replace("\n","") 
        
len (fulldocument)        
#create the list of sentences from full documnet
 

doc=nlp(fulldocument)

sentences=[str(sent) for sent in doc.sents]
# del doc
sentences[3900]
count=0
for i in sentences:
    count+=1
 
count

#create the full text without unnecessary sencentes , just keep sentences with at least one instance of medicine in them
fultext_clean=''
count=0
for d in list_of_names:
    print(d)
    for sent in sentences:
        if   str(sent.lower()).find(' '+ d.lower()) != -1 or str(sent.lower()).find('/'+ d.lower()) != -1 or str(sent.lower()).find('\\'+ d.lower()) != -1:
            fultext_clean += sent + "\n"
            print(d)
            count+=1
           
            
len(fultext_clean)        
           
    
newpath= f'{path}\\corpus_maker\\_21_singular_sentences.txt'
file = open( newpath, "w",encoding="utf8") 
file.write(fultext_clean) 
file.close()  

#remove douplicate lines in the final ducument
with open(f'{path}\\corpus_maker\\_21_singular_sentences.txt',encoding="utf8") as result:
        uniqlines = set(result.readlines())
        with open(f'{path}\\corpus_maker\\_21_singular_sentences_cleaned.txt', 'w',encoding="utf8") as rmdup:
            rmdup.writelines(set(uniqlines))
            



            
#functoins to create tuples of entities, it creartes a tuple for each entity in the sentence
with open(f'{path}\\corpus_maker\\_21_singular_sentences_cleaned.txt',encoding="utf8") as fc:            
  text_cleand = fc.read()          
            
def find_all_indexes(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


nlp2 = spacy.load('en_core_web_sm')
doc2=nlp(text_cleand)

sentences_C=[str(sent) for sent in doc2.sents]
del doc
del doc2



Large_corpus=[] 

for sent in sentences_C:
    sent=sent.replace("\n","")
   # print(len(sent))
    if len(sent)>10 and len(sent)<2000:
        sent_tuple=()
        dict_in_sentence={}
        list_in_sentence=[]
       
    
        for i in range(0, len(list_of_names)):
            if  list_of_names[i] =='ICS' and ('cholinergics' in  sent.lower()   or  'muscarinic' in  sent.lower()):
                 continue
            else:
                starts= find_all_indexes(sent.lower(),str(list_of_names[i].lower()+'s'))
                            
                for sta in starts:
                    tup=(sta, 1 + sta + entity_length[i],entity_label[i])
                    list_in_sentence.append(tup) 
                                        
         
                
                starts_s= find_all_indexes(sent.lower(),str(list_of_names[i].lower()))
                starts_s=[x for x in starts_s if x not in starts]
                #print(starts_s)  
                for sta_s in starts_s:
                    tup_s=(sta_s,  sta_s+ entity_length[i] ,entity_label[i])
                    list_in_sentence.append(tup_s)
                        
         
        
        dict_in_sentence['entities'] = list_in_sentence 
        sent_tuple=(sent,dict_in_sentence)
        print(sent_tuple)  
        
        Large_corpus.append(sent_tuple)
        
 
     

with open(f'{path}\\corpus_maker\\_22_tupels.ICS.txt', 'w',encoding="utf8") as filehandle:
    filehandle.writelines("%s\n" % str(tup) for tup in Large_corpus)
    
 

    