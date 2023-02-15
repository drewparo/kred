import ast
from sklearn.decomposition import PCA
import requests
from operator import itemgetter
from utils import extract_wikidata
import csv

# Function to create Job dataset
def create_jobs_dataset(records, jobs_info):
    djob = {}
    wikidata_ids = []
    with open('datasets/LinkedIn-Tech-Job-Data/jobs.csv', 'a', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=jobs_info)
        for job in records:
            djob["post_id"] = job['post_id']
            djob["industries"] = job['Industries']
            djob["job_function"] = job['Job function']
            djob["title"] = job['title']
            djob["abstract"] = extract_wikidata.extract_significant_sentences(job['description']) ##
            djob["post_url"] = job['post_url']
            djob["entity_info_title"] = extract_wikidata.get_extract(job['description'], wikidata_ids)
            djob["entity_info_abstract"] = []
            writer.writerow(djob)

    # Creation file with all uniques wikidata_ids
    file = open('datasets/LinkedIn-Tech-Job-Data/wikidata_ids.txt', 'w', encoding='utf-8')
    for wikidata in wikidata_ids:
        file.write(wikidata + "\n")
    file.close()
    return wikidata_ids

# Function to get entity description
def get_description(id):
    query = f"""
    SELECT ?Label ?Description
    WHERE 
    {{
      wd:{id} rdfs:label ?Label .
      FILTER (LANG(?Label) = "en").
      OPTIONAL {{ wd:{id} schema:description ?Description . FILTER (LANG(?Description) = "en") }}
    }}
    """
    max_tries = 100
    for i in range(max_tries):
      try:
        response = requests.get("https://query.wikidata.org/sparql", params={'query': query, 'format': 'json'})
        response_json = response.json()
        label = response_json['results']['bindings'][0]['Label']['value']
        description = response_json['results']['bindings'][0].get('Description', {}).get('value', '')
        description = label + ' ' + description
        return description
      except:
        pass
    return None


# Function to extract the embeddings for a given entity
def extract_embeddings_entities(entity_id, model):
    entity_description = get_description(entity_id)
    if entity_description == None:
        return None, None
    sentence_embeddings = model.encode(entity_description)
    return entity_description, sentence_embeddings

# Function to extract the embedding for a given description
def extract_embeddings_relation(relation_id, model):
    relation_description = get_description(relation_id)
    if relation_description == None:
        return None, None
    sentence_embeddings = model.encode(relation_description)
    return relation_description, sentence_embeddings

# Extract entity embeddings and descriptions
def extract_entity_embeddings_descriptions(entity_ids,model):
    entities_embeddings = []
    entities_descriptions = {}
    for entity_id in entity_ids:
        entity_description, sentence_embeddings = extract_embeddings_entities(entity_id, model)
        if not entity_description is None:
            entities_embeddings.append(sentence_embeddings)
            entities_descriptions[entity_id] = entity_description
    return entities_embeddings, entities_descriptions

# Extract relation embeddings and descriptions
def extract_relation_embeddings_descriptions(relation_ids,model):
    relation_embeddings = []
    relation_descs = []
    for relation_id in relation_ids:
        rela_d, emb_ = extract_embeddings_relation(relation_id, model)
        if not emb_ is None:
            relation_embeddings.append(emb_)
            relation_descs.append(rela_d)
    return relation_embeddings, relation_descs

# Function to reduce the size of the embeddings
def reduce_embeddings_size(embeddings):
    pca = PCA(n_components=100)
    return pca.fit_transform(embeddings)

# Function to remove not embedded entities form jobs
def update_jobs_dataset(df_jobs,descripriptions):
    new_col = []
    entities_good = list(descripriptions.keys())
    for r in df_jobs['entity_info_title']:
        entities = ast.literal_eval(r)
        new_entities = []
        for e in entities:
            entity = ast.literal_eval(e)
            if entity['WikidataId'] in entities_good:
                entity['OccurrenceOffsets'] = [entity['OccurrenceOffsets']]
                new_entities.append(entity)
        new_col.append(new_entities)
    
    df_jobs['entity_info_title'] = new_col
    return df_jobs

# Function to create entities/relation to id variables
def create_x2id_dict(elements):
    result = {}
    for e in elements:
        result[e] = len(result)

    return result 

# Extract relationships between entities
def extract_relationships(entity_ids):
    # Define the API endpoint for retrieving information about entities
    endpoint = "https://www.wikidata.org/w/api.php"
    # Split entities in chuncks to foolow wikidata api constraints
    chunks = [entity_ids[x:x+50] for x in range(0, len(entity_ids), 50)]

    entities = {}

    for c in chunks:
    # Define the parameters for the API request
        params = {
            "action": "wbgetentities",
            "ids": "|".join(c),
            "format": "json"
        }

        # Send the API request and retrieve the response
        response = requests.get(endpoint, params=params)

        # Extract the JSON data from the response
        data = response.json()
        #print(data)
        

        # Extract the entity information from the data
        for entity_id, entity in data["entities"].items():
            #print(entity_id)
            try:
                entities[entity_id] = {
                    "label": entity["labels"]["en"]["value"],
                    "description": entity["descriptions"]["en"]["value"],
                    "claims": entity.get("claims", {})
                }
            except:
                entities[entity_id] = {
                    "label": entity["labels"]["en"]["value"],
                    "description": entity["labels"]["en"]["value"],
                    "claims": entity.get("claims", {})
                }

    # Define a list to store the relationships between entities
    relationships = []

    # Extract the relationships between entities from the entity information
    for entity_id, entity in entities.items():
        for property_id, property_values in entity["claims"].items():
            for property_value in property_values:
                if "mainsnak" in property_value and "datavalue" in property_value["mainsnak"]:
                    datavalue = property_value["mainsnak"]["datavalue"]
                    if "value" in datavalue and "id" in datavalue["value"]:
                        #print(datavalue)
                        try:
                            target_entity_id = datavalue["value"]["id"]
                            if target_entity_id in entity_ids:
                                relationships.append((entity_id, property_id, target_entity_id))
                        except:
                            pass
    
    return relationships

# Function to create entity to id
def get_triple2id(relationships,entity2id_dict,relation2id_dict):
    relationships = list(set(relationships))
    triple2id = [(entity2id_dict[relation[0]],entity2id_dict[relation[2]],relation2id_dict[relation[1]]) for relation in relationships]
    triple2id = sorted(triple2id, key=itemgetter(0, 1, 2))
    return triple2id