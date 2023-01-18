from tqdm import tqdm
import os
import json


def news_title(adressa_path):
    hash_title = {}
    for file in os.scandir(adressa_path):
        with open(file, "r") as f:
            for line in tqdm(f):
                event_dict = json.loads(line.strip("\n"))
                if "id" in event_dict and "title" in event_dict:
                    hash_title[event_dict["id"]] = event_dict["title"]
        break  # Only one file
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

        news_line = "\t".join([str(nindex), "", "", title, "", "", "", ""]) + "\n"
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
            subcategory = article_content['category1'].split('|')[-1]
        except:
            continue

        news_line = "\t".join([str(nindex), category, subcategory, title, "", "", "", ""]) + "\n"
        news_lines.append(news_line)

    for stage in ["train", "valid", "test"]:
        file_path = out_path / stage
        file_path.mkdir(exist_ok=True, parents=True)
        with open(out_path / stage / "news.tsv", "w", encoding="utf-8") as f:
            f.writelines(news_lines)
