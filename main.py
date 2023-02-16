from utils.jobs_processing import *
from utils.user_processing import *
from utils.util_jobs import *

def main_preprocessing(config):
    task = input("Write the type of task you want to execute. Choose between user2item, item2item, vert_classify, pop_predict or multi-task: ")
    epochs = input("Write the number of epochs. Suggestion: 5 or 10 epochs: ")
    batch_size = input("Write the size of the batch. Suggested: 32 or 64: ")
    if task.lower() == "user2item" or task.lower() =='item2item' or  task.lower() =="vert_classify" or task.lower() =="pop_predict":
        train_type= "single_task"
    elif task.lower() == "multi-task":
        train_type = task.lower()

    config['trainer']['epochs'] = epochs
    config['data_loader']['batch_size'] = batch_size
    config['trainer']['training_type'] = train_type
    config['trainer']['task'] = task

    # Read LinkedIn dataset
    df_jobs = pd.read_csv('/datasets/LinkedIn-Tech-Job-Data/jobs.csv')
    records = df_jobs.to_dict(orient='records')
    jobs_info = ['post_id', 'industries', 'job_function', 'title', 'abstract', 'post_url', 'entity_info_title',
                     'entity_info_abstract']
    # Generate jobs dataset for KRED
    wikidata_ids = create_jobs_dataset(records, jobs_info)

    # Define model for sentence embeddings
    model = SentenceTransformer('all-mpnet-base-v2').to('cuda:0')

    # Extract entity embeddings and descriptions
    entity_embeddings, entity_descriptions = extract_entity_embeddings_descriptions(wikidata_ids,model)

    # Reduce the embeddings size
    reduced_entity_embeddings = reduce_embeddings_size(entity_embeddings)

    # Save entity embeddings file
    np.savetxt('datasets/LinkedIn-Tech-Job-Data/entity2vecd100_jobs.vec', reduced_entity_embeddings, fmt='%.6f', delimiter='\t')

    # Update jobs removing not embedded entities
    df_jobs = update_jobs_dataset(df_jobs,entity_descriptions)

    # List oh embedded entities
    entities = list(entity_descriptions.keys())

    # Create entity-to-id dictionary
    entity2id_dict = create_x2id_dict(entities)

    # Save entity2id file
    with open('datasets/LinkedIn-Tech-Job-Data/entity2id_jobs.txt', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for key, value in entity2id_dict.items():
            writer.writerow([key, value])
            writer.writerow([key, value])

    # Extract relationships between entities
    relationships = extract_relationships(entities)

    # Extract unique relations
    relations = list(set([rel[1] for rel in relationships]))

    # Extract relation embeddings and relations
    relations_embeddings, relations_descriptions = extract_relation_embeddings_descriptions(relations,model) ####

    # Reduce the embeddings size
    reduced_embeddings_relations = reduce_embeddings_size(relations_embeddings)

    # Save relation embedding file
    np.savetxt('datasets/LinkedIn-Tech-Job-Data/relation2vecd100_jobs.vec', reduced_embeddings_relations, fmt='%.6f', delimiter='\t')

    # Create relation to id
    relation2id_dict = create_x2id_dict(list(relations_descriptions.keys()))

    # Save relation2id file
    with open('datasets/LinkedIn-Tech-Job-Data/relation2id_jobs.txt', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for key, value in relation2id_dict.items():
            writer.writerow([key, value])
            writer.writerow([key, value])

    # Generate entity-entity-relation triples
    triple2id = get_triple2id(relationships,entity2id_dict,relation2id_dict)

    # Save triple to id file
    with open('datasets/LinkedIn-Tech-Job-Data/triple2id_jobs.txt', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for tuple_ in triple2id:
            writer.writerow(tuple_)

    # Update jobs extracting entities per job and removing unuseful columns
    df_jobs = modify_df_jobs(df_jobs)

    #From stackoverflow survey, extract wikidata_ids

    user_info = ["User", "Entities"]
    df = pd.read_csv('datasets/LinkedIn-Tech-Job-Data/clean_stackoverflow_survey.csv')

    # Generate csv for user, entities in survey
    fix_survey(df, user_info)

    df_users = pd.read_csv('/datasets/LinkedIn-Tech-Job-Data/user_entities_stackoverflow.csv')

    # Generate history and behaviors gpr
    df_users = generate_history_behaviors(df_users, df_jobs)

    df_users.to_csv("datasets/LinkedIn-Tech-Job-Data/behaviors_jobs.tsv", sep='\t')

    #  Split behaviors in train and validation
    split_behaviors('datasets/LinkedIn-Tech-Job-Data/behaviors_jobs.tsv')