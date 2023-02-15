import pandas as pd
import csv
from jobs_processing import *
from user_processing import *
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

# Read LinkedIn dataset
df_jobs = pd.read_csv('jobs.csv')
records = df_jobs.to_dict(orient='records')

# Generate jobs dataset for KRED
jobs_dict,wikidata_ids = create_jobs_dataset(records)

# Save jobs file
with open('jobs.csv', 'a', encoding='utf-8', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = list(jobs_dict.keys()))

# Define model for sentence embeddings
model = SentenceTransformer('all-mpnet-base-v2').to('cuda:0')

# Extract entity embeddings and descriptions
entity_embeddings, entity_descriptions = extract_entity_embeddings_descriptions(wikidata_ids,model)

# Reduce the embeddings size
reduced_entity_embeddings = reduce_embeddings_size(entity_embeddings)

# Save entity embeddings file
np.savetxt('entity2vecd100_jobs.vec', reduced_entity_embeddings, fmt='%.6f', delimiter='\t')

# Update jobs removing not embedded entities
df_jobs = update_jobs_dataset(df_jobs,entity_descriptions)

# List oh embedded entities
entities = list(entity_descriptions.keys())

# Create entity-to-id dictionary
entity2id_dict = create_x2id_dict(entities)

# Save entity2id file
with open('entity2id.txt', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for key, value in entity2id_dict.items():
        writer.writerow([key, value])
        writer.writerow([key, value])

# Extract relationships between entities
relationships = extract_relationships(entities)

# Extract unique relations
relations = list(set([rel[1] for rel in relationships]))

# Extract relation embeddings and relations
relations_embeddings, relations_descriptions = extract_relation_embeddings_descriptions(relations,model)

# Reduce the embeddings size
reduced_embeddings_relations = reduce_embeddings_size(relations_embeddings)

# Save relation embedding file
np.savetxt('relation2vecd100_jobs.vec', reduced_embeddings_relations, fmt='%.6f', delimiter='\t')

# Create relation to id 
relation2id_dict = create_x2id_dict(list(relations_descriptions.keys()))

# Save relation2id file
with open('relation2id.txt', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for key, value in relation2id_dict.items():
        writer.writerow([key, value])
        writer.writerow([key, value])

# Generate entity-entity-relation triples
triple2id = get_triple2id(relationships,entity2id_dict,relation2id_dict)

# Save triple to id file
with open('triple2id_jobs.txt', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    for tuple_ in triple2id:
        writer.writerow(tuple_)

# Update jobs extracting entities per job and removing unuseful columns
df_jobs = modify_df_jobs(df_jobs)

##### Read asd process user files

#df_users=user_entities_so

# Generate history and behaviors gpr
df_jobs = generate_history_behaviors(df_jobs,df_users)


