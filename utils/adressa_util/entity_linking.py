from refined.inference.processor import Refined
from transformers import M2M100ForConditionalGeneration
from utils.tokenization_small100 import SMALL100Tokenizer
from refined.data_types.base_types import Entity, Span
import json
from tqdm import tqdm
import os

refined = Refined.from_pretrained(
    model_name='wikipedia_model',
    entity_set="wikidata"
)
model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
tokenizer.tgt_lang = "en"
# Remeber to cite https://huggingface.co/alirezamsh/small100

def spans_to_mind_format(results: Span):
    entity_list = []
    for entity in results:
        if entity.predicted_entity.wikidata_entity_id == None:
            continue
            # An entity has been detected but does not have a wikidata page
        entity_list.append({
            'Label': entity.predicted_entity.wikipedia_entity_title,
            'Type': entity.coarse_mention_type,
            'WikidataId': entity.predicted_entity.wikidata_entity_id,
            'Confidence': entity.candidate_entities[0][1],
            'OccurrenceOffsets': [entity.start],
            'SurfaceForms': [entity.text]
        })
    return entity_list


def ned_no(text):
    encoded_no_ = tokenizer(text, return_tensors="pt")
    generated_tokens_ = model.generate(**encoded_no_)
    en_text_ = tokenizer.batch_decode(generated_tokens_, skip_special_tokens=True)[0]
    list_entity = spans_to_mind_format(refined.process_text(en_text_))
    list_entity_json = json.dumps(list_entity)
    return list_entity_json


def getEncodedNews(file_path_new):
    encoded_id = set()
    if not os.path.exists(file_path_new):
        return {}
    with open(file_path_new, "r") as file1:
        for line in tqdm(file1):
            attributes = line.split('\t')
            id_ = attributes[0]
            encoded_id.add(id_)
    return encoded_id


def add_entities(out_path):
    file_path_new = out_path / 'train' / 'news_new.tsv'
    global tokenizer, model, refined
    for stage in ["train"]:
        # check if existing encoded news are present
        encoded_news_id = getEncodedNews(out_path / stage / 'news_new.tsv')
        file_path = out_path / stage / 'news.tsv'
        with open(file_path_new, "a") as new_file:
            with open(file_path, "r") as original_file_news:
                for line in tqdm(original_file_news):
                    attributes = line.split('\t')
                    title = attributes[3]
                    abstract = attributes[4]
                    title_ned = ned_no(title)  #  NED for title, NO -> EN -> NED -> RETURN LIST
                    abstract_ned = ned_no(abstract)  #  NED for title, NO -> EN -> NED -> RETURN LIST
                    attributes[6] = title_ned
                    attributes[7] = abstract_ned
                    new_file.write("\t".join(attributes))
                    new_file.write('\n')
