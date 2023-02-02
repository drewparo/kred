from refined.inference.processor import Refined
from transformers import M2M100ForConditionalGeneration
from utils.tokenization_small100 import SMALL100Tokenizer
from refined.data_types.base_types import Entity, Span
import json

refined = Refined.from_pretrained(
    model_name='wikipedia_model',
    entity_set="wikidata"
)
model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
tokenizer.tgt_lang = "en"


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


def add_entities(out_path):
    global tokenizer, model, refined
    for stage in ["train", "valid", "test"]:
        file_path = out_path / stage / 'news.tsv'
        with open(file_path, "r") as file1:
            for line in file1:
                attributes = line.split('\t')
                title = attributes[3]
                encoded_no_title = tokenizer(title, return_tensors="pt")
                generated_tokens_title = model.generate(**encoded_no_title)
                en_text_title = tokenizer.batch_decode(generated_tokens_title, skip_special_tokens=True)[0]
                list_entity = spans_to_mind_format(refined.process_text(en_text_title))
                list_entity_json = json.dumps(list_entity)

