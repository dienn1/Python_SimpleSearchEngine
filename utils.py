import json
import os


# Read a chunk from a partial index
def read_chunk(file, chunk_size):
    res_dict = dict()
    lines = file.readlines(chunk_size)
    for line in lines:
        data = json.loads(line)
        data_dict = {data[0]: data[1]}
        res_dict.update(data_dict)
    return res_dict if len(res_dict) > 0 else None


def initialize_directory(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    path = os.path.join(root_dir, "partial_indexes")
    if os.path.exists(path):
        return
    os.mkdir(path)


def dump_json(obj, path):
    with open(path, "w") as f:
        json_dump = json.dumps(obj, indent=4)
        f.write(json_dump)


def load_json(path):
    if not os.path.exists(path):
        print(path, "does not exist")
        return
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def dump_jsonl(dict_obj, path):
    if not isinstance(dict_obj, dict):
        raise TypeError("dict_obj is not of Dictionary type")
    with open(path, "w") as outfile:
        for i in dict_obj.items():
            item_dump = json.dumps(i)
            outfile.write(item_dump)
            outfile.write("\n")


# dump dict to jsonl sorted by key
def dump_jsonl_sorted(dict_obj, path):
    if type(dict_obj) is not type(dict()):
        raise TypeError("dict_obj is not of Dictionary type")
    with open(path, "w") as outfile:
        for key, value in sorted(dict_obj.items(), key=lambda item: int(item[0])):
            item_dump = json.dumps([key, value])
            outfile.write(item_dump)
            outfile.write("\n")


def load_jsonl(path):
    if not os.path.exists(path):
        print(path, "does not exist")
        return
    dict_obj = dict()
    with open(path, "r") as file:
        for line in file:
            data = json.loads(line)
            data_dict = {data[0]: data[1]}
            dict_obj.update(data_dict)
    return dict_obj