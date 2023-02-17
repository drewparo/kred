import requests
import nltk
import spacy
import ast
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from txtai.pipeline import Summary
from nltk.corpus import stopwords
import spacy
from heapq import nlargest
nlp = spacy.load('en_core_web_sm')

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
STOP_WORDS = set(stopwords.words('english'))
stopwords = list(STOP_WORDS)
summary = Summary()


def get_ner(text):
    sent = nltk.word_tokenize(text)
    entities = nltk.ne_chunk(nltk.pos_tag(sent))
    return entities

def get_wikidata_id(entity):
    query = entity.split()
    query = "+".join(query)
    response = requests.get(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item&limit=1&search={query}")
    if response.status_code == 200:
        response_json = response.json()
        if "search" in response_json and response_json["search"]:
            return response_json["search"][0]["id"]
        else:
            return None
    return None

def extract_significant_sentences(text):
    try:
        global summary, punctuation
        if text is None or text == '':
            return ''
        outcome = text
        text_length = len(text)
        doc = nlp(text)
        tokens = [token.text for token in doc]
        punctuation = punctuation + '\n'
        word_frequencies = {}

        for word in doc:
            if word.text.lower() not in stopwords:
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        max_frequency = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency
        sentence_tokens = [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

        select_length = int(len(sentence_tokens) * 0.5)

        summary1 = nlargest(select_length, sentence_scores,
                            key=sentence_scores.get)
        sumup = summary(summary1[0].text)
        return sumup
    except Exception as e:
        print('error')
        return text


def get_extract(text, wikidata_ids):
  try:
      t = nlp(text)
      list_dict = []
      keys = ["Label", "WikidataId", "OccurrenceOffsets", "Type"]
      wiki_h = []
      entities = get_ner(text)
      for entity in entities.subtrees():
          wikidata_dict = {}
          #print(entity.label())
          if entity.label() != 'S':
            entity_name = " ".join([i[0] for i in entity.leaves()])
            wikidata_id = get_wikidata_id(entity_name)
            if wikidata_id and wikidata_id not in wiki_h:
                if wikidata_id not in wikidata_ids:
                    wikidata_ids.append(wikidata_id)
                wiki_h.append(wikidata_id)
                wikidata_dict["Label"] = entity_name
                wikidata_dict["WikidataId"] = wikidata_id

                for token in t:
                    if str(token) == entity_name:
                        wikidata_dict["OccurrenceOffsets"] = token.idx
                        break
                for ent in t.ents:
                    if ent.text in wikidata_dict["Label"]:
                        wikidata_dict["Type"] = ent.label_[0]
                if wikidata_dict and all(key in wikidata_dict for key in keys):
                    list_dict.append(json.dumps(wikidata_dict, sort_keys=True))
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
