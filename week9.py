#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:50:26 2021

@author: hakyungsung
"""

def conllu_dicter(text,splitter="\t"):
	output_list = [] #list for each sentence and token

	sents = text.split("\n\n") #split text into sentences

	for sent in sents:
		if sent == "":
			continue
		sent_anno = [] #sentence list for tokens
		lines = sent.split("\n")
		for line in lines:
			if len(line) == 0:
				continue
			token = {}
			if line[0] == "#": #skip lines without target annotation
				continue

			anno = line.split(splitter) #split by splitting character (in the case of our example, this will be a tab character)

			#now, we will grab all relevant information and add it to our token dictionary:
			#token["idx"] = anno[0]
			token["word"] = anno[1]#get word
			token["upos"] = anno[3] #get the universal pos tag
			#token["xpos"] = anno[4] #get the xpos tag(s) - in English this is usally Penn tags
			#token["dep"] = anno[7] #dependency relationship
			#token["head_idx"] = anno[6] #id of dependency head
			sent_anno.append(token) #append token dictionary to sentence level list
		output_list.append(sent_anno) #append sentence level list to
	return(output_list)

def tupler(lolod,posname = "pos"):
    outlist = []
    for sent in lolod: #iterate through sentences
        outsent = []
        for token in sent: #iterate through tokens
            outsent.append((token["word"], token[posname])) #create tuples
        outlist.append(outsent)    
    return(outlist) #return list of lists of tuples

from simple_perceptron import PerceptronTagger as SimpleTron
from full_perceptron import PerceptronTagger as FullTron

# strip tags if necessary, apply tagger

def test_tagger(test_sents,model,tag_strip = False, word_loc = 0):
    
    if tag_strip == True:
        sent_words = []
        for sent in test_sents:
                ws = []
                for token in sent:
                        ws.append(token[word_loc])
                sent_words.append(ws)
                
    else:
            sent_words = test_sents
    tagged_sents = []
    
    for sent in sent_words:
            tagged_sents.append(model.tag(sent))
            
    return(tagged_sents)

def simple_accuracy_sent(gold,test):
# gold: hand-tagged list, test: machine-tagged text
        correct = 0 #correct count
        nwords = 0 #total words count
        
        for sent_id, sents in enumerate(gold):
                for word_id, (gold_word, gold_tag) in enumerate(sents):
                        nwords += 1
                        if gold_tag == test[sent_id][word_id][1]:
                                correct += 1
        return(correct/nwords)
    

korean_train = tupler(conllu_dicter(open("ko_kaist-ud-train.conllu.txt").read()),posname = "upos")
korean_test = tupler(conllu_dicter(open("ko_kaist-ud-test.conllu.txt").read()),posname = "upos")

tagger_ko = SimpleTron(load=False) #define tagger

tagger_ko.train(korean_train,save_loc = "small_feature_Kaist_train_perceptron.pickle") #train tagger on train_data, save the model as "small_feature_Browntrain_perceptron.pickle"

#load pretrained model (if needed)
tagger_ko = SimpleTron(load = True, PICKLE = "small_feature_Kaist_train_perceptron.pickle")


tagged_ko_test = test_tagger(korean_test,tagger_ko,tag_strip = True)
print(simple_accuracy_sent(tagged_ko_test,korean_test)) #0.847846012832264


tagger2_ko = FullTron(load=False)
tagger2_ko.train(korean_train,save_loc = "full_feature_Kaist_train_perceptron.pickle")

tagged_ko_test2 = test_tagger(korean_test,tagger2_ko,tag_strip = True)

simple_accuracy_sent(korean_test,tagged_ko_test2)