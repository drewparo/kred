import requests
import nltk
import spacy
import ast
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
nlp = spacy.load("en_core_web_sm")
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
def get_ner(text):
    sent = nltk.word_tokenize(text)
    entities = nltk.ne_chunk(nltk.pos_tag(sent))
    return entities

def get_wikidata_id(entity):
    query = entity.split()
    query = "+".join(query)
    response = requests.get(f'https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&limit=1&search={query}')
    if response.status_code == 200:
        response_json = response.json()
        if 'search' in response_json and response_json['search']:
            return response_json['search'][0]['id']
        else:
            return None
    return None

def extract_significant_sentences(text):
    try:
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # Create a dictionary to store the importance of each sentence
        sentence_importance = {}

        # Loop through each sentence and calculate its importance based on word frequency
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            frequency = nltk.FreqDist(words)
            sentence_importance[sentence] = frequency.most_common(1)[0][1]

        # Sort the sentences by their importance
        sorted_sentences = sorted(sentence_importance.items(), key=lambda x: x[1], reverse=True)

        # Return the most important sentence
        return sorted_sentences[0][0]
    except TypeError:
        return ""

def get_extract(text, wikidata_ids):
  try:
      t = nlp(text)
      list_dict = []
      #wikidata_ids = []
      wiki_h = []
      entities = get_ner(text)
      for entity in entities.subtrees():
          wikidata_dict = {}
          #print(entity.label())
          if entity.label() != 'S':
            entity_name = " ".join([i[0] for i in entity.leaves()])
            #print(entity_name)
            wikidata_id = get_wikidata_id(entity_name)
            if wikidata_id and wikidata_id not in wiki_h:
                if wikidata_id not in wikidata_ids:
                    wikidata_ids.append(wikidata_id)
                wiki_h.append(wikidata_id)
                wikidata_dict['Label'] = entity_name
                wikidata_dict['WikidataId'] = wikidata_id
                for token in t:
                    if str(token) == entity_name: ##to modify
                        wikidata_dict['OccurrenceOffsets'] = token.idx
                        break
                list_dict.append(wikidata_dict)
      del wiki_h
      return list_dict
  except ValueError:
      return {}


def get_ids(col_dict, wikidata_ids):
    for d in col_dict:
        row = ast.literal_eval(d)
        for i in row:
            if i['WikidataId'] not in wikidata_ids:
                wikidata_ids.append(i['WikidataId'])
