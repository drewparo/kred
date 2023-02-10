import json
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from utils.adressa_util.UserInfo import UserInfo


def news_title(adressa_path, content_news):
    article_present = set()
    for file in os.scandir(content_news):
        article_present.add(file.name)

    hash_title = {}

    for file in os.scandir(adressa_path):
        with open(file, "r") as f:
            for line in tqdm(f):
                event_dict = json.loads(line.strip("\n"))
                if "id" in event_dict and "title" in event_dict and event_dict['id'] in article_present:
                    hash_title[event_dict["id"]] = event_dict["title"]
    hash2id = {k: v for k, v in zip(hash_title.keys(), range(1, len(hash_title) + 1))}
    return hash_title, hash2id


"""
News ID	N37378
Category	sports
SubCategory	golf
Title	PGA Tour winners
Abstract	A gallery of recent winners on the PGA Tour.
URL	https://www.msn.com/en-us/sports/golf/pga-tour-winners/ss-AAjnQjj?ocid=chopendata
Title Entities	[{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["PGA Tour"]}]
Abstract Entites	[{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [35], "SurfaceForms": ["PGA Tour"]}]
"""


def write_news_files(news_title, nid2index, out_path):
    # Output with MIND format
    news_lines = []
    for nid in tqdm(news_title):
        nindex = nid2index[nid]
        title = news_title[nid]

        news_line = "\t".join(['N'+str(nindex), "", "", title, "", "", "", ""]) + "\n"
        news_lines.append(news_line)

    for stage in ["train", "valid", "test"]:
        file_path = out_path / stage
        file_path.mkdir(exist_ok=True, parents=True)
        with open(out_path / stage / "news.tsv", "w", encoding="utf-8") as f:
            f.writelines(news_lines)


def write_news_files_full(news_title, nid2index, out_path, articles_content):
    # Output with MIND format
    news_lines = []
    for nid in tqdm(news_title):
        nindex = nid2index[nid]
        title = news_title[nid]
        try:
            article_content = articles_content[nid]
            category = article_content['category0']
            if 'category1' in article_content:
                subcategory = article_content['category1'].split('|')[-1]
            else:
                subcategory = category
            if 'description' in article_content:
                abstract = article_content['description']
            elif 'teaser' in article_content:
                abstract = article_content['teaser'].replace('\t', ' ').replace('\n', ' ')
            else:
                abstract = title
            url = article_content['url']
        except Exception as e:
            print(e)

            continue

        news_line = "\t".join(['N' + str(nindex), category, subcategory, title, abstract, url, "", ""]) + "\n"
        news_lines.append(news_line)

    for stage in ["train", "valid", "test"]:
        file_path = out_path / stage
        file_path.mkdir(exist_ok=True, parents=True)
        with open(out_path / stage / "news.tsv", "w", encoding="utf-8") as f:
            f.writelines(news_lines)


def entities_news(news_path):
    # Function that return all entities in news file

    entities = set()
    # Read entities from train news
    with open(news_path) as fp:

        for line in fp:
            try:
                newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split(
                    '\t')
                # Add entities in the title
                for entity in eval(
                        entity_info_title):  # Eval from string to data type eg "{1,2,3}" return a set with elements 1,2,3
                    entities.add(entity["WikidataId"])
                # Add entities in the abstract
                for entity in eval(entity_info_abstract):
                    entities.add(entity["WikidataId"])
            except:
                print(line)
    return entities


def process_users(adressa_path, nid2index, content_news):
    article_present = set()
    for file in os.scandir(content_news):
        article_present.add(file.name)
    uid2index = {}
    user_info = defaultdict(UserInfo)

    for file in os.scandir(adressa_path):
        with open(file, "r") as f:
            for l in tqdm(f):
                event_dict = json.loads(l.strip("\n"))
                if "id" in event_dict and "title" in event_dict and event_dict['id'] in article_present:
                    nindex = nid2index[event_dict["id"]]
                    uid = event_dict["userId"]

                    if uid not in uid2index:
                        uid2index[uid] = len(uid2index)

                    uindex = uid2index[uid]
                    click_time = int(event_dict["time"])
                    day = int(file.name[-1])
                    user_info[uindex].update(nindex, click_time, day)

    return uid2index, user_info


def construct_behaviors(hash_title, uindex, click_news, train_news, test_news, neg_num):
    train_lines = []
    test_lines = []
    p = np.ones(len(hash_title) + 1, dtype="float32")
    p[click_news] = 0
    p[train_news] = 0
    p[test_news] = 0
    p[0] = 0
    p /= p.sum()

    train_his_news = ['N'+str(i) for i in click_news.tolist()]
    train_his_line = " ".join(train_his_news)
    # Return trainline
    for nindex in train_news:
        neg_cand = np.random.choice(
            len(hash_title) + 1, size=neg_num, replace=False, p=p
        ).tolist()
        cand_news = " ".join(
            [f"N{str(nindex)}-1"] + [f"N{str(nindex)}-0" for nindex in neg_cand]
        )

        train_behavior_line = f"null\t{uindex}\tnull\t{train_his_line}\t{cand_news}\n"
        train_lines.append(train_behavior_line)

    test_his_news = ['N'+str(i) for i in click_news.tolist() + train_news.tolist()]
    test_his_line = " ".join(test_his_news)
    for nindex in test_news:
        neg_cand = np.random.choice(
            len(hash_title) + 1, size=neg_num, replace=False, p=p
        ).tolist()
        cand_news = " ".join(
            [f"N{str(nindex)}-1"] + [f"N{str(nindex)}-0" for nindex in neg_cand]
        )

        test_behavior_line = f"null\t{uindex}\tnull\t{test_his_line}\t{cand_news}\n"
        test_lines.append(test_behavior_line)
    return train_lines, test_lines
