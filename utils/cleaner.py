import shutil
from utils.util_new import *


def cleaner(config):
    change_in_train = False
    change_in_val = False
    entity2id_dict = entity_to_id(config, entities_news(config))
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    with open(config["data"]["train_news"]) as old, open('tmp/train_' + config["data"]["train_news"].split('/')[-1],
                                                         'w+') as new:
        for line in old:
            newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split(
                '\t')
            entity_info_title = json.loads(entity_info_title)
            for index, entity in enumerate(entity_info_title):
                if entity2id_dict.get(entity['WikidataId'], config["data"]["num_entity_embedding"]) > config["data"][
                    "num_entity_embedding"]:
                    change_in_train = True
                    del entity_info_title[index]
            entity_info_title = json.dumps(entity_info_title)
            entity_info_abstract = json.loads(entity_info_abstract)
            for index, entity in enumerate(entity_info_abstract):
                if entity2id_dict.get(entity['WikidataId'], config["data"]["num_entity_embedding"]) > config["data"][
                    "num_entity_embedding"]:
                    change_in_train = True
                    del entity_info_abstract[index]
            entity_info_abstract = json.dumps(entity_info_abstract)
            new_line = '\t'.join([newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract])
            new.write(new_line + '\n')
    with open(config["data"]["valid_news"]) as old, open('tmp/test_' + config["data"]["valid_news"].split('/')[-1],
                                                         'w+') as new:
        for line in old:
            newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract = line.strip().split(
                '\t')
            entity_info_title = json.loads(entity_info_title)
            for index, entity in enumerate(entity_info_title):
                if entity2id_dict.get(entity['WikidataId'], config["data"]["num_entity_embedding"]) > config["data"][
                    "num_entity_embedding"]:
                    change_in_val = True
                    del entity_info_title[index]
            entity_info_title = json.dumps(entity_info_title)
            entity_info_abstract = json.loads(entity_info_abstract)
            for index, entity in enumerate(entity_info_abstract):
                if entity2id_dict.get(entity['WikidataId'], config["data"]["num_entity_embedding"]) > config["data"][
                    "num_entity_embedding"]:
                    change_in_val = True
                    del entity_info_abstract[index]
            entity_info_abstract = json.dumps(entity_info_abstract)
            new_line = '\t'.join([newsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract])
            new.write(new_line + '\n')
    try:
        if change_in_val:
            src_test = 'tmp/test_' + config["data"]["valid_news"].split('/')[-1]
            dst_test = '/'.join(config["data"]["valid_news"].split('/')[:-1]) + '/'
            shutil.copy(src_test, dst_test)
        if change_in_train:
            src_train = 'tmp/train_' + config["data"]["train_news"].split('/')[-1]
            dst_train = '/'.join(config["data"]["train_news"].split('/')[:-1]) + '/'
            shutil.copy(src_train, dst_train)
        return config
    except Exception as e:
        print(e)
        print('Replacing name in config')
        config['data']['train_news'] = 'tmp/train_' + config["data"]["train_news"].split('/')[-1]
        config['data']['valid_news'] = 'tmp/test_' + config["data"]["valid_news"].split('/')[-1]
        return config
