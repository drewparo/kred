import shutil
import csv
import pandas as pd
from utils.util_jobs import *

def escape_quote(string):
    result = []
    n = 0
    while n < len(string):
        if n >=2 and string[n].isalnum() and (string[n-2] == ':' or string[n-1] == '{' or string[n+1] == ',' or string[n+1] == '}'):
            if (string[n-2] == ':' or string[n-1] == '{') and (string[n+1] == ',' or string[n+1] == '}'):
                result.append('"')
                result.append(string[n])
                result.append('"')
            elif string[n-2] == ':' or string[n-1] == '{':
                result.append('"')
                result.append(string[n])
            else:
                result.append(string[n])
                result.append('"')
        else:
            if string[n] == "'" and \
                    (n == 0 or string[n - 1] != '\\') and \
                    (n == len(string) - 1 or string[n + 1] != "'") and \
                    not (string[n-1].isalnum() and string[n+1].isalnum()):
                result.append('"')
            else:
                result.append(string[n])
        n += 1
    return "".join(result)
def cleaner_jobs(config):

    change_in_train = False
    change_in_val = False
    entity2id_dict = entity_to_id_jobs(config, entities_jobs(config))
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    with open(config["jobs"]["train_jobs"], encoding='utf-8') as old, open('tmp/' + config["jobs"]["train_jobs"].split('/')[-1],'w+', encoding='utf-8') as new:
        # Create a csv reader object
        reader_old = csv.reader(old, delimiter='\t')

        # Skip the first line
        next(reader_old)

        for line in reader_old:
            jobsid = line[0]
            vert = line[1]
            subvert = line[2]
            title = line[3]
            abstract = line[4]
            url = line[5]
            entity_info_title = escape_quote(str(line[6]))
            entity_info_abstract = line[7]
            entity_info_title = json.loads(entity_info_title)

            for index, entity in enumerate(entity_info_title):
                if not entity_info_title:
                    break
                if entity2id_dict.get(entity['WikidataId'], config["jobs"]["num_entity_embedding"]) > config["jobs"][
                    "num_entity_embedding"]:
                    change_in_train = True
                    del entity_info_title[index]
            entity_info_title = json.dumps(entity_info_title)
            #entity_info_abstract = json.loads(entity_info_abstract)
            #for index, entity in enumerate(entity_info_abstract):
            #    if entity2id_dict.get(entity['WikidataId'], config["jobs"]["num_entity_embedding"]) > config["jobs"][
            #        "num_entity_embedding"]:
            #        change_in_train = True
            #        del entity_info_abstract[index]
            #entity_info_abstract = json.dumps(entity_info_abstract)
            new_line = '\t'.join([jobsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract])
            new.write(new_line + '\n')

    with open(config["jobs"]["valid_jobs"], encoding='utf-8') as old, open('tmp/' + config["jobs"]["valid_jobs"].split('/')[-1], 'w+', encoding='utf-8') as new:
        # Create a csv reader object
        reader_old = csv.reader(old, delimiter='\t')

        # Skip the first line
        next(reader_old)

        for line in reader_old:
            jobsid = line[0]
            vert = line[1]
            subvert = line[2]
            title = line[3]
            abstract = line[4]
            url = line[5]
            entity_info_title = escape_quote(str(line[6]))
            entity_info_abstract = line[7]
            # print(entity_info_title)
            entity_info_title = json.loads(entity_info_title)
            for index, entity in enumerate(entity_info_title):
                if entity2id_dict.get(entity['WikidataId'], config["jobs"]["num_entity_embedding"]) > config["jobs"][
                    "num_entity_embedding"]:
                    change_in_val = True
                    del entity_info_title[index]
            entity_info_title = json.dumps(entity_info_title)
            entity_info_abstract = json.loads(entity_info_abstract)
            for index, entity in enumerate(entity_info_abstract):
                if entity2id_dict.get(entity['WikidataId'], config["jobs"]["num_entity_embedding"]) > config["jobs"][
                    "num_entity_embedding"]:
                    change_in_val = True
                    del entity_info_abstract[index]
            entity_info_abstract = json.dumps(entity_info_abstract)
            new_line = '\t'.join([jobsid, vert, subvert, title, abstract, url, entity_info_title, entity_info_abstract])
            new.write(new_line + '\n')

    try:
        if change_in_val:
            src_test = 'tmp/test_' + config["jobs"]["valid_jobs"].split('/')[-1]
            dst_test = '/'.join(config["jobs"]["valid_jobs"].split('/')[:-1]) + '/'
            shutil.copy(src_test, dst_test)
        if change_in_train:
            src_train = 'tmp/train_' + config["jobs"]["train_jobs"].split('/')[-1]
            dst_train = '/'.join(config["jobs"]["train_jobs"].split('/')[:-1]) + '/'
            shutil.copy(src_train, dst_train)
        return config
    except:
        print('Replacing name in config')
        config['jobs']['train_jobs'] = os.path.abspath('tmp/test_' + config["jobs"]["valid_jobs"].split('/')[-1])
        config['jobs']['valid_jobs'] = os.path.abspath('tmp/train_' + config["jobs"]["train_jobs"].split('/')[-1])
        return config
