import numpy as np
import ast

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

def generate_history_behaviors(df_users,df_jobs,p):
    clicks_col = []
    behaviors_col = []
    for user_entities in df_users['entities']:
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

    df_users = df_users[['user','Histories','Behaviors']]
    return df_users