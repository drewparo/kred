import numpy as np
import ast
import csv

import pandas as pd

from utils import extract_wikidata
def modify_df_jobs(df_jobs):
    new_col = []
    for r in df_jobs['entity_info_title']:
        present = []
        entities = ast.literal_eval(r)
        for entity in entities:
            present.append(entity['WikidataId'])
        new_col.append(present)
    df_jobs['entities'] = new_col
    df_jobs = df_jobs[['post_id','entities']]
    return df_jobs

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

def fix_survey(df, user_info):
    with open('datasets/LinkedIn-Tech-Job-Data/user_entities_stackoverflow.csv', 'w', encoding='utf-8',
              newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=user_info)
        writer.writeheader()
        djob = {}
        for i, job in zip(range(0, 10001), df):  # truncated to 10000
            wikidata_ids = []
            djob["User"] = "U" + str(i)
            language = extract_wikidata.get_extract(df.Language[i], wikidata_ids)
            database = extract_wikidata.get_extract(df.Database[i], wikidata_ids)
            platform = extract_wikidata.get_extract(df.Platform[i], wikidata_ids)
            webframe = extract_wikidata.get_extract(df.Webframe[i], wikidata_ids)
            misctech = extract_wikidata.get_extract(df.MiscTech[i], wikidata_ids)
            toolstech = extract_wikidata.get_extract(df.ToolsTech[i], wikidata_ids)
            opsys = extract_wikidata.get_extract(df.OpSys[i], wikidata_ids)
            # Save the couple U0, ['Q123', 'Q456', ... ]
            djob["Entities"] = wikidata_ids
            # Write the file in order to save them once for all
            writer.writerow(djob)


def generate_history_behaviors(df_users,df_jobs):
    p = 0.75
    clicks_col = []
    behaviors_col = []
    for user_entities in df_users['Entities']:
        feasible_jobs = [] 
        clicks = []
        behaviors = []
        for index, row in df_jobs.iterrows():
            n_occurrences = 0
            for entity in ast.literal_eval(user_entities):
                if entity in row['entities']:
                    n_occurrences += 1
            if n_occurrences > 0:
                feasible_jobs.append((row['post_id'],n_occurrences))

        probabilities = softmax([i[1] for i in feasible_jobs])

        if len(feasible_jobs) > 0:
            if set([i[0] for i in feasible_jobs]) == set(clicks+[j.split('-')[0] for j in behaviors]):
                break

            while True: 
                new_click = np.random.choice([i[0] for i in feasible_jobs], p=probabilities)
                if new_click not in clicks:
                    clicks.append(new_click)
                                
                if len(clicks) >= 3:
                    if np.random.choice([0,1], p=[p,1-p]):
                        break

                if set([i[0] for i in feasible_jobs]) == set(clicks+[j.split('-')[0] for j in behaviors]):
                    break

            while len(behaviors) < 20:
                new_beh = np.random.choice(list(df_jobs['post_id'].values))
                if new_beh not in clicks:
                    if new_beh in [i[0] for i in feasible_jobs]:
                        behaviors.append(new_beh+'-1')
                    else:
                        behaviors.append(new_beh+'-0')

        clicks_col.append(clicks)
        behaviors_col.append(behaviors)
    
    clicks_str = [' '.join(r) for r in clicks_col]
    behaviors_str = [' '.join(r) for r in behaviors_col]
    df_users['Histories'] = clicks_str
    df_users['Behaviors'] = behaviors_str

    df_users = df_users[['User','Histories','Behaviors']]
    return df_users

def split_behaviors(filename):
    with open(filename, 'r', encoding='utf-8') as csv_file:
        # Load the data from the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file, sep='\t')
        df = df.drop(df.columns[:1], axis=1)

        # Calculate the number of rows in the DataFrame
        n = df.shape[0]

        # Generate a random permutation of the indices
        idx = np.random.permutation(n)

        # Split the indices into training and validation sets
        split = int(0.8 * n)
        train_idx = idx[:split]
        val_idx = idx[split:]

        # Split the DataFrame into training and validation sets based on the indices
        train_df = df.iloc[train_idx, :]
        val_df = df.iloc[val_idx, :]

        # Save the training and validation sets as CSV files
        train_df.to_csv('datasets/LinkedIn-Tech-Job-Data/behaviors_train_jobs.tsv', sep='\t', index=False)
        val_df.to_csv('datasets/LinkedIn-Tech-Job-Data/behaviors_valid_jobs.tsv', sep='\t', index=False)
