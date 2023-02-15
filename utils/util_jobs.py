import gzip
import json
import pandas as pd
import torch
import random
import numpy as np
import os
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from IPython.core.display_functions import clear_output
from sentence_transformers import SentenceTransformer
import requests
import math
import zipfile
from tqdm import tqdm
import pickle


# Create file and save data using pickle
def save_to_pickle(data, file_name):
    with open(file_name, "wb") as fp:
        pickle.dump(data, fp)



# Load data from file using pickle
def load_from_pickle(filename):
    with open(filename, "rb") as fp:
        data = pickle.load(fp)
    return data

def entities_jobs(config):
    entities = set()
    # Read entities from train jobs config["data"]["train_jobs"]=jobs.csv
    with open(config['jobs']['wikidata_ids'], 'r', encoding='utf-8') as file:
        entities = set(line.strip() for line in file)
    return entities

# Return a dictionary from entity names to ids
def entity_to_id_jobs(config, entities):
    entity2id_dict = {}
    # Get the association entity-id from the file
    with open(config["jobs"]["entity_index"], encoding='utf-8') as fp:
        #entity_num = int(fp.readline().split('\n')[0])
        for line in fp:
            entity, entityid = line.strip().split('\t')
            if entity in entities:
                # Entity id is increased by one in order to be compatible with all the following operations
                entity2id_dict[entity] = int(entityid) + 1
    return entity2id_dict


# Return a dictionary from entity ids to names
def id_to_entity_jobs(config, ids):
    entity2id_dict = {}
    # Get the association entity-id from the file
    with open(config["jobs"]["entity_index"], encoding='utf-8') as fp:
        #entity_num = int(fp.readline().split('\n')[0])
        for line in fp:
            entity, entityid = line.strip().split('\t')
            # Since the entity ids are increased by one when reading from the file,
            # then it is also done here before the comparison
            if int(entityid) + 1 in ids:
                entity2id_dict[entity] = int(entityid) + 1
    return entity2id_dict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def construct_adj(graph_file, entity2id_file, args):  # graph is triple
    print('constructing adjacency matrix jobs ...')
    graph_file_fp = open(graph_file, 'r', encoding='utf-8')
    graph = []
    for line in graph_file_fp:
        linesplit = line.split('\n')[0].split('\t')
        if len(linesplit) > 1:
            graph.append([linesplit[0], linesplit[2], linesplit[1]])

    kg = {}
    for triple in graph:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

    fp_entity2id = open(entity2id_file, 'r', encoding='utf-8')
    entity_num = int(fp_entity2id.readline().split('\n')[0])
    entity_adj = []
    relation_adj = []
    for i in range(entity_num):
        entity_adj.append([])
        relation_adj.append([])
    for key in kg.keys():
        for index in range(args.entity_neighbor_num):
            i = random.randint(0, len(kg[key]) - 1)
            entity_adj[int(key)].append(int(kg[key][i][0]))
            relation_adj[int(key)].append(int(kg[key][i][1]))
    entity_adj = np.array(entity_adj)
    relation_adj = np.array(relation_adj)
    return entity_adj, relation_adj


def construct_embedding(entity_embedding_file, relation_embedding_file):
    print('constructing embedding ...')
    entity_embedding = []
    relation_embedding = []
    fp_entity_embedding = open(entity_embedding_file, 'r', encoding='utf-8')
    fp_relation_embedding = open(relation_embedding_file, 'r', encoding='utf-8')
    for line in fp_entity_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        entity_embedding.append(linesplit)
    for line in fp_relation_embedding:
        linesplit = line.strip().split('\t')
        linesplit = [float(i) for i in linesplit]
        relation_embedding.append(linesplit)
    return torch.FloatTensor(entity_embedding), torch.FloatTensor(relation_embedding)


def my_collate_fn(batch):
    return batch


def construct_entity_dict(entity_file):
    fp_entity2id = open(entity_file, 'r', encoding='utf-8')
    entity_dict = {}
    entity_num_all = int(fp_entity2id.readline().split('\n')[0])
    lines = fp_entity2id.readlines()
    for line in lines:
        entity, entityid = line.strip().split('\t')
        entity_dict[entity] = entityid
    return entity_dict


def real_batch(batch):
    data = {}
    data['item1'] = []
    data['item2'] = []
    data['label'] = []
    for item in batch:
        data['item1'].append(item['item1'])
        data['item2'].append(item['item2'])
        data['label'].append(item['label'])
    return data


def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
            ):
                file.write(data)
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


def unzip_file(zip_src, dst_dir, clean_zip_file=True):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)


def get_user2item_data_jobs(config):
    negative_num = config['trainer']['train_neg_num']
    train_data = {}
    user_id = []
    news_id = []
    label = []
    fp_train = open(config['jobs']['train_behavior'], 'r', encoding='utf-8')
    next(fp_train)
    for line in fp_train:
        userid, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        positive_list = []
        negative_list = []
        for news in behavior:
            newsid, news_label = news.split('-')
            if news_label == "1":
                positive_list.append(newsid)
            else:
                negative_list.append(newsid)
        for pos_news in positive_list:
            user_id.append(userid + "_train")
            if len(negative_list) >= negative_num:
                neg_news = random.sample(negative_list, negative_num)
            else:
                neg_news = negative_list
                for i in range(negative_num - len(negative_list)):
                    neg_news.append("N0")
            all_news = neg_news
            all_news.append(pos_news)
            news_id.append(all_news)
            label.append([])
            for i in range(negative_num):
                label[-1].append(0)
            label[-1].append(1)

    train_data['item1'] = user_id
    train_data['item2'] = news_id
    train_data['label'] = label

    dev_data = {}
    session_id = []
    user_id = []
    news_id = []
    label = []
    fp_dev = open(config['jobs']['valid_behavior'], 'r', encoding='utf-8')
    next(fp_dev)

    for index, line in enumerate(fp_dev):
        userid, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        for news in behavior:
            newsid, news_label = news.split('-')
            session_id.append(index)
            user_id.append(userid + "_dev")
            if news_label == "1":
                news_id.append(newsid)
                label.append(1.0)
            else:
                news_id.append(newsid)
                label.append(0.0)

    dev_data['item1'] = user_id
    dev_data['session_id'] = session_id
    dev_data['item2'] = news_id
    dev_data['label'] = label

    return train_data, dev_data

def build_user_history_jobs(config):
    user_history_dict = {}
    fp_train_behavior = open(config['jobs']['train_behavior'], 'r', encoding='utf-8')
    for line in fp_train_behavior:
        user_id, history, behavior = line.strip().split('\t')
        if len(history.split(' ')) >= config['model']['user_his_num']:
            user_history_dict[user_id + "_train"] = history.split(' ')[:config['model']['user_his_num']]
        else:
            user_history_dict[user_id + "_train"] = history.split(' ')
            for i in range(config['model']['user_his_num'] - len(history.split(' '))):
                user_history_dict[user_id + "_train"].append("N0")
            if user_history_dict[user_id + "_train"][0] == '':
                user_history_dict[user_id + "_train"][0] = 'N0'

    fp_dev_behavior = open(config['jobs']['valid_behavior'], 'r', encoding='utf-8')
    for line in fp_dev_behavior:
        user_id, history, behavior = line.strip().split('\t')
        if len(history.split(' ')) >= config['model']['user_his_num']:
            user_history_dict[user_id + "_dev"] = history.split(' ')[:config['model']['user_his_num']]
        else:
            user_history_dict[user_id + "_dev"] = history.split(' ')
            for i in range(config['model']['user_his_num'] - len(history.split(' '))):
                user_history_dict[user_id + "_dev"].append("N0")
            if user_history_dict[user_id + "_dev"][0] == '':
                user_history_dict[user_id + "_dev"][0] = 'N0'
    return user_history_dict

def build_news_features_mind_jobs(config, entity2embedding_dict, embedding_folder=None):
    jobs_features = {}

    jobs_feature_dict = {}
    fp_train_jobs = open(config['jobs']['train_jobs'], 'r', encoding='utf-8')
    df = pd.read_csv(fp_train_jobs, sep='\t')
    df.replace(np.nan,0)
    for index, row in df.iterrows():
        jobsid = row['post_id']
        title = row['title']
        abstract = str(row['abstract'])
        entity_info_title = eval(row['entity_info_title'])
        entity_info_abstract = []
        jobs_feature_dict[jobsid] = (title + " " + abstract, entity_info_title, entity_info_abstract)

    fp_dev_jobs = open(config['jobs']['valid_jobs'], 'r', encoding='utf-8')
    df = pd.read_csv(fp_dev_jobs, sep='\t')
    df.replace(np.nan, 0)
    for index, row in df.iterrows():
        jobsid = row['post_id']
        title = row['title']
        abstract = str(row['abstract'])
        entity_info_title = eval(row['entity_info_title'])
        entity_info_abstract = []
        jobs_feature_dict[jobsid] = (title + " " + abstract, entity_info_title, entity_info_abstract)

    entity_type_dict = {}
    entity_type_index = 1
    # Load sentence embeddings from files if present
    if embedding_folder is not None:
        sentences_embedding = load_from_pickle(embedding_folder + "train_jobs_embeddings")
        sentences_embedding.extend(load_from_pickle(embedding_folder + "valid_jobs_embeddings"))
    else:
        model = SentenceTransformer('all-mpnet-base-v2')

    for i, jobs in enumerate(jobs_feature_dict):
        if embedding_folder is not None:
            sentence_embedding = sentences_embedding[i]
        else:
            sentence_embedding = model.encode(jobs_feature_dict[jobs][0])
            clear_output()
        jobs_entity_feature_list = []
        title_entity_json = json.dumps(jobs_feature_dict[jobs][1])
        jobs_entity_feature = {}
        for item in json.loads(title_entity_json):
            if item['Type'] not in entity_type_dict:
                entity_type_dict[item['Type']] = entity_type_index
                entity_type_index = entity_type_index + 1
            occ_off = [item['OccurrenceOffsets']]
            jobs_entity_feature[item['WikidataId']] = (len(occ_off), 1, entity_type_dict[
                item['Type']])
        for entity in jobs_entity_feature:
            if entity in entity2embedding_dict:
                jobs_entity_feature_list.append(
                    [entity2embedding_dict[entity], jobs_entity_feature[entity][0], jobs_entity_feature[entity][1],
                     jobs_entity_feature[entity][2]])
        jobs_entity_feature_list.append([0, 0, 0, 0])
        if len(jobs_entity_feature_list) > config['model']['news_entity_num']:
            jobs_entity_feature_list = jobs_entity_feature_list[:config['model']['news_entity_num']]
        else:
            for i in range(len(jobs_entity_feature_list), config['model']['news_entity_num']):
                jobs_entity_feature_list.append([0, 0, 0, 0])
        jobs_feature_list_ins = [[], [], [], [], []]
        for i in range(len(jobs_entity_feature_list)):
            for j in range(4):
                jobs_feature_list_ins[j].append(jobs_entity_feature_list[i][j])
        jobs_feature_list_ins[4] = sentence_embedding
        jobs_features[jobs] = jobs_feature_list_ins
    jobs_features["N0"] = [[], [], [], [], []]
    for i in range(config['model']['news_entity_num']):
        for j in range(4):
            jobs_features["N0"][j].append(0)
    jobs_features["N0"][4] = np.zeros(config['model']['document_embedding_dim'])
    return jobs_features, 100, 10, 100


def construct_adj_mind_jobs(config, entity2id_dict, entity2embedding_dict):  # graph is triple
    print('constructing adjacency matrix ...')
    entities_ids = set(entity2id_dict.values())
    with open(config['jobs']['knowledge_graph'], 'r', encoding='utf-8') as graph_file_fp:
        kg = {}
        for line in graph_file_fp:
            linesplit = line.split('\n')[0].split('\t')
            head = int(linesplit[0]) + 1
            relation = int(linesplit[2]) + 1
            tail = int(linesplit[1]) + 1
            # treat the KG as an undirected graph
            # Restrict only to the selected entities and their relations
            if head in entities_ids:
                if head not in kg:
                    kg[head] = []
                kg[head].append((tail, relation))
            if tail in entities_ids:
                if tail not in kg:
                    kg[tail] = []
                kg[tail].append((head, relation))

    entity_num = len(entity2embedding_dict)
    entity_adj = []
    relation_adj = []
    id2entity_dict = {v: k for k, v in entity2id_dict.items()}
    for i in range(entity_num + 1):
        entity_adj.append([])
        relation_adj.append([])
    for i in range(config['model']['entity_neighbor_num']):
        entity_adj[0].append(0)
        relation_adj[0].append(0)
    for key in kg.keys():
        for index in range(config['model']['entity_neighbor_num']):
            i = random.randint(0, len(kg[key]) - 1)
            # Convert the id to the new one
            new_key = entity2embedding_dict[id2entity_dict[int(key)]]
            entity_adj[new_key].append(int(kg[key][i][0]))
            relation_adj[new_key].append(int(kg[key][i][1]))

    print('construct_adj_mind finish')
    return entity_adj, relation_adj


# Load the emdedding of the entities in entity2id_dict, append them to entity_embedding and
# update entity2embedding_dict
def construct_embedding_jobs(config, entity2id_dict, entity_embedding, entity2embedding_dict):
    print('constructing embedding jobs...')
    relation_embedding = []
    zero_array = np.zeros(config['model']['entity_embedding_dim'])
    relation_embedding.append(zero_array)
    id2entity_dict = {v: k for k, v in entity2id_dict.items()}
    with open(config['jobs']['entity_embedding'], 'r', encoding='utf-8') as fp_entity_embedding:
        i = 1
        for line in fp_entity_embedding:
            if i in id2entity_dict:
                linesplit = line.strip().split('\t')
                linesplit = [float(i) for i in linesplit]
                entity2embedding_dict[id2entity_dict[i]] = len(entity_embedding)
                entity_embedding.append(linesplit)
            i += 1
    with open(config['jobs']['relation_embedding'], 'r', encoding='utf-8') as fp_relation_embedding:
        for line in fp_relation_embedding:
            linesplit = line.strip().split('\t')
            linesplit = [float(i) for i in linesplit]
            relation_embedding.append(linesplit)
    return entity2embedding_dict, entity_embedding, relation_embedding


def build_vert_data_jobs(config):
    random.seed(2023)
    vert_label_dict = {}
    label_index = 0
    all_jobs_data = []
    vert_train = {}
    vert_dev = {}
    item1_list_train = []
    item2_list_train = []
    label_list_train = []
    item1_list_dev = []
    item2_list_dev = []
    label_list_dev = []
    fp_train_jobs = open(config['jobs']['train_jobs'], 'r', encoding='utf-8')
    df = pd.read_csv(fp_train_jobs, sep='\t')
    df.replace(np.nan, 0)
    for index, row in df.iterrows():
        jobsid = row['post_id']
        vert = row['industries']
        if vert not in vert_label_dict:
            vert_label_dict[vert] = label_index
            label_index = label_index + 1
        all_jobs_data.append((jobsid, vert_label_dict[vert]))
    for i in range(len(all_jobs_data)):
        if random.random() < 0.8:
            item1_list_train.append("U0")
            item2_list_train.append(all_jobs_data[i][0])
            label_list_train.append(all_jobs_data[i][1])
        else:
            item1_list_dev.append("U0")
            item2_list_dev.append(all_jobs_data[i][0])
            label_list_dev.append(all_jobs_data[i][1])
    vert_train['item1'] = item1_list_train
    vert_train['item2'] = item2_list_train
    vert_train['label'] = label_list_train
    vert_dev['item1'] = item1_list_dev
    vert_dev['item2'] = item2_list_dev
    vert_dev['label'] = label_list_dev

    return vert_train, vert_dev


def build_pop_data_jobs(config):
    fp_train = open(config['jobs']['train_behavior'], 'r', encoding='utf-8')
    news_imp_dict = {}
    pop_train = {}
    pop_test = {}
    next(fp_train)
    for line in fp_train:
        userid, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        for news in behavior:
            newsid, news_label = news.split('-')
            if news_label == "1":
                if newsid not in news_imp_dict:
                    news_imp_dict[newsid] = [1, 1]
                else:
                    news_imp_dict[newsid][0] = news_imp_dict[newsid][0] + 1
                    news_imp_dict[newsid][1] = news_imp_dict[newsid][1] + 1
            else:
                if newsid not in news_imp_dict:
                    news_imp_dict[newsid] = [0, 1]
                else:
                    news_imp_dict[newsid][1] = news_imp_dict[newsid][1] + 1
    return pop_train, pop_test


def build_item2item_data_jobs(config):
    fp_train = open(config['jobs']['train_behavior'], 'r', encoding='utf-8')
    next(fp_train)
    item2item_train = {}
    item2item_test = {}
    item1_train = []
    item2_train = []
    label_train = []
    item1_dev = []
    item2_dev = []
    label_dev = []
    user_history_dict = {}
    jobs_click_dict = {}
    doc_doc_dict = {}
    all_jobs_set = set()
    for line in fp_train:
        userid, history, behavior = line.strip().split('\t')
        behavior = behavior.split(' ')
        if userid not in user_history_dict:
            user_history_dict[userid] = set()
        for news in behavior:
            newsid, news_label = news.split('-')
            all_jobs_set.add(newsid)
            if news_label == "1":
                user_history_dict[userid].add(newsid)
                if newsid not in jobs_click_dict:
                    jobs_click_dict[newsid] = 1
                else:
                    jobs_click_dict[newsid] = jobs_click_dict[newsid] + 1
        news = history.split(' ')
        for newsid in news:
            user_history_dict[userid].add(newsid)
            if newsid not in jobs_click_dict:
                jobs_click_dict[newsid] = 1
            else:
                jobs_click_dict[newsid] = jobs_click_dict[newsid] + 1
    for user in user_history_dict:
        list_user_his = list(user_history_dict[user])
        for i in range(len(list_user_his) - 1):
            for j in range(i + 1, len(list_user_his)):
                doc1 = list_user_his[i]
                doc2 = list_user_his[j]
                if doc1 != doc2:
                    if (doc1, doc2) not in doc_doc_dict and (doc2, doc1) not in doc_doc_dict:
                        doc_doc_dict[(doc1, doc2)] = 1
                    elif (doc1, doc2) in doc_doc_dict and (doc2, doc1) not in doc_doc_dict:
                        doc_doc_dict[(doc1, doc2)] = doc_doc_dict[(doc1, doc2)] + 1
                    elif (doc2, doc1) in doc_doc_dict and (doc1, doc2) not in doc_doc_dict:
                        doc_doc_dict[(doc2, doc1)] = doc_doc_dict[(doc2, doc1)] + 1
    weight_doc_doc_dict = {}
    for item in doc_doc_dict:
        if item[0] in jobs_click_dict and item[1] in jobs_click_dict:
            weight_doc_doc_dict[item] = doc_doc_dict[item] / math.sqrt(
                jobs_click_dict[item[0]] * jobs_click_dict[item[1]])

    THRED_CLICK_TIME = 10
    freq_news_set = set()
    for news in jobs_click_dict:
        if jobs_click_dict[news] > THRED_CLICK_TIME:
            freq_news_set.add(news)
    news_pair_thred_w_dict = {}  # {(new1, news2): click_weight}
    for item in weight_doc_doc_dict:
        if item[0] in freq_news_set and item[1] in freq_news_set:
            news_pair_thred_w_dict[item] = weight_doc_doc_dict[item]

    news_positive_pairs = []
    for item in news_pair_thred_w_dict:
        if news_pair_thred_w_dict[item] > 0.05:
            news_positive_pairs.append(item)

    for item in news_positive_pairs:
        random_num = random.random()
        if random_num < 0.8:
            item1_train.append(item[0])
            item2_train.append(item[1])
            label_train.append(1)
            negative_list = random.sample(list(freq_news_set), 4)
            for negative in negative_list:
                item1_train.append(item[0])
                item2_train.append(negative)
                label_train.append(0)
        else:
            item1_dev.append(item[0])
            item2_dev.append(item[1])
            label_dev.append(1)
            negative_list = random.sample(list(freq_news_set), 4)
            for negative in negative_list:
                item1_train.append(item[0])
                item2_train.append(negative)
                label_dev.append(0)
    item2item_train["item1"] = item1_train
    item2item_train["item2"] = item2_train
    item2item_train["label"] = label_train
    item2item_test["item1"] = item1_dev
    item2item_test["item2"] = item2_dev
    item2item_test["label"] = label_dev
    return item2item_train, item2item_test


def load_data_mind_jobs(config, embedding_folder=None):
    # Build the dictionary for all the entities in the news
    entity2id_dict = entity_to_id_jobs(config, entities_jobs(config))

    # Initialize the list containing all the embeddings
    entity_embedding = []
    # Ids start from 1, so append a list of zeros at index 0
    entity_embedding.append(np.zeros(config['model']['entity_embedding_dim']))
    # Initialize the dictionary mapping the entity name and the position of embedding in the list
    entity2embedding_dict = {}
    # Load only the embeddings of the entities in the news
    entity2embedding_dict, entity_embedding, relation_embedding = construct_embedding_jobs(config, entity2id_dict,
                                                                                           entity_embedding,
                                                                                           entity2embedding_dict)

    # For each entity in the news, get the neighbours entities in the wikidata graph and their relations
    entity_adj, relation_adj = construct_adj_mind_jobs(config, entity2id_dict, entity2embedding_dict)

    for index in range(len(entity_adj)):
        if len(entity_adj[index]) == 0:
            entity_adj[index] = [0 for i in range(config['model']['entity_neighbor_num'])]
    for index in range(len(relation_adj)):
        if len(relation_adj[index]) == 0:
            relation_adj[index] = [0 for i in range(config['model']['entity_neighbor_num'])]

    # Some of the entities in the neighborhood are not part of the entities in the news, so after having identiefied them,
    # their embedding is added to the list and they are added to the dictionary of entity2embedding
    entities_not_embedded = set([item for items in entity_adj for item in items]).difference(
        set(entity2id_dict.values()))
    entity2id_dict_not_embedded = id_to_entity_jobs(config, entities_not_embedded)
    entity2embedding_dict, entity_embedding, relation_embedding = construct_embedding_jobs(config,
                                                                                           entity2id_dict_not_embedded,
                                                                                           entity_embedding,
                                                                                           entity2embedding_dict)

    # Add the new entities to the dictionary
    entity2id_dict.update(entity2id_dict_not_embedded)
    # Invert the dictionary
    id2entity_dict = {v: k for k, v in entity2id_dict.items()}
    id2entity_dict[0] = 'Q87'

    # The ids in entity_adj are the original ones, they need to be updated to the new ids given by entity2embedding_dict
    for i in range(1, len(entity_adj)):
        for j in range(0, len(entity_adj[i])):
            entity_adj[i][j] = entity2embedding_dict[id2entity_dict[entity_adj[i][j]]]
    entity_embedding = torch.FloatTensor(np.array(entity_embedding))
    relation_embedding = torch.FloatTensor(np.array(relation_embedding))

    # Load the news
    news_feature, max_entity_freq, max_entity_pos, max_entity_type = build_news_features_mind_jobs(config,
                                                                                              entity2embedding_dict,
                                                                                              embedding_folder)

    # Load the user history
    user_history = build_user_history_jobs(config)

    if config['trainer']['training_type'] == "multi-task":
        train_data, dev_data = get_user2item_data_jobs(config)
        vert_train, vert_test = build_vert_data_jobs(config)
        pop_train, pop_test = build_pop_data_jobs(config)
        item2item_train, item2item_test = build_item2item_data_jobs(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data, vert_train, vert_test, pop_train, pop_test, item2item_train, item2item_test
    elif config['trainer']['task'] == "user2item":
        train_data, dev_data = get_user2item_data_jobs(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data
    elif config['trainer']['task'] == "item2item":
        item2item_train, item2item_test = build_item2item_data_jobs(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, item2item_train, item2item_test
    elif config['trainer']['task'] == "vert_classify":
        vert_train, vert_test = build_vert_data_jobs(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, vert_train, vert_test
    elif config['trainer']['task'] == "pop_predict":
        pop_train, pop_test = build_pop_data_jobs(config)
        return user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, pop_train, pop_test
    else:
        print("task error, please check config")


def load_compressed_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_compressed_pickle(filename, obj):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pretrained_data_mind_jobs(config):
    data_path = config['jobs']['jobs_data']
    if data_path:
        restored_data = load_compressed_pickle(config['jobs']['jobs_data'])
        user_history = restored_data["user_history"]
        entity_embedding = restored_data["entity_embedding"]
        relation_embedding = restored_data["relation_embedding"]
        entity_adj = restored_data["entity_adj"]
        relation_adj = restored_data["relation_adj"]
        news_feature = restored_data["jobs_feature"]
        max_entity_freq = restored_data["max_entity_freq"]
        max_entity_pos = restored_data["max_entity_pos"]
        max_entity_type = restored_data["max_entity_type"]
        train_data = restored_data["train_data"]
        dev_data = restored_data["dev_data"]
        vert_train = restored_data["vert_train"]
        vert_test = restored_data["vert_test"]
        pop_train = restored_data["pop_train"]
        pop_test = restored_data["pop_test"]
        item2item_train = restored_data["item2item_train"]
        item2item_test = restored_data["item2item_test"]
        if config['trainer']['training_type'] == "multi-task":
            data = user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data, vert_train, vert_test, pop_train, pop_test, item2item_train, item2item_test
        elif config['trainer']['task'] == "user2item":
            data = user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, train_data, dev_data
        elif config['trainer']['task'] == "item2item":
            data = user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, item2item_train, item2item_test
        elif config['trainer']['task'] == "vert_classify":
            data = user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, vert_train, vert_test
        elif config['trainer']['task'] == "pop_predict":
            data = user_history, entity_embedding, relation_embedding, entity_adj, relation_adj, news_feature, max_entity_freq, max_entity_pos, max_entity_type, pop_train, pop_test
        return data
    else:
        print(f"Data not found at {data_path}")
        return None

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
