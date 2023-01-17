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
