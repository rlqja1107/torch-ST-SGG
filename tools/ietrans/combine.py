import json, pickle
import numpy as np
import sys, os
from tqdm import tqdm
# extra: ['img_path', 'boxes', 'labels', 'pairs', 'possible_rels', 'rel_logits']
# intra: ['width', 'height', 'img_path', 'boxes', 'labels', 'relations', 'possible_rels', 'logits']

exp_dir = os.environ.get("EXP")
code_dir = os.environ.get("SG")
validate_internal = False
validate_external = True

extra_data = pickle.load(open("External/External_1.0.pk", "rb")) if validate_external else pickle.load(open("External/External_1.0.pk", "rb"))
intra_data = pickle.load(open("Internal/Yes_internal_trans_0.7", "rb")) if validate_internal else pickle.load(open("Internal/No_internal_trans", "rb"))
img_infos = json.load(open("datasets/vg_50/image_data.json", "r"))
img_infos = {k["image_id"]: k for k in img_infos}

# filter_out relations in extra
freq_rels = [0, 31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43, 40, 49, 41, 23]
# freq_rels = {r: 0 for r in freq_rels}
freq_rels = np.array(freq_rels, dtype=np.int64)

def get_img_id(path):
    return int(path.split("/")[-1].replace(".jpg", ""))


def to_dic(data):
    dic = {}
    for d in data:
        dic[get_img_id(d['img_path'])] = d
    return dic


def complete_img_info(data, img_infos):
    imid = get_img_id(data["img_path"])
    iminfo = img_infos[imid]
    data["width"] = iminfo["width"]
    data["height"] = iminfo["height"]


def filter_out_freq_rels(relations):
    rels = relations[:, 2]
    in_mask = (rels.reshape(-1, 1) == freq_rels.reshape(-1, 1).T).any(-1)
    return relations[~in_mask]


def complete_relatinos(data):
    assert "rel_logits" in data, data
    rel_logits = data["rel_logits"]
    possible_rels = data["possible_rels"]
    pairs = data["pairs"]
    rels = []
    for poss_rls, logits in zip(possible_rels, rel_logits):
        max_id = logits.argmax()
        rels.append(poss_rls[max_id])
    rels = np.array(rels, dtype=pairs.dtype).reshape(-1, 1)
    rels = np.concatenate([pairs, rels], 1)
    data["relations"] = filter_out_freq_rels(rels)
    del data["pairs"]
    return data


def ex_data_to_in_data(ex_data, img_infos):
    complete_img_info(ex_data, img_infos)
    complete_relatinos(ex_data)
    del ex_data["rel_logits"]
    return ex_data


def merge_ex_and_in(ex_data, in_data):
    assert ex_data["img_path"] == in_data["img_path"]
    if validate_external:
        ex_data = ex_data_to_in_data(ex_data, img_infos)
        ex_rels = ex_data["relations"]
        ex_pairs = ex_rels[:, :2]
    else:
        ex_rels = np.array([], dtype=np.int64)
        ex_rels.shape = (0,3)
        ex_pairs = ex_rels[:, :2]
    in_rels = in_data["relations"]
    in_pairs = in_rels[:, :2]
    intersect_rels = ex_pairs.reshape(ex_pairs.shape[0], ex_pairs.shape[1], 1) \
                     == in_pairs.reshape(in_pairs.shape[0], in_pairs.shape[1], 1).T
    intersect_rels = intersect_rels.all(-1).any(1)
    unintersect_rels = ~intersect_rels
    final_rels = np.concatenate([in_rels, ex_rels[unintersect_rels]], 0)
    in_data["relations"] = final_rels
    return in_data

# print("intra", len(intra_data), sum([len(x["relations"]) for x in intra_data]) )

# to dic
# extra: ['img_path', 'boxes', 'labels', 'pairs', 'possible_rels', 'rel_logits']
# intra: ['width', 'height', 'img_path', 'boxes', 'labels', 'relations', 'possible_rels', 'logits']
extra_data = to_dic(extra_data)
intra_data = to_dic(intra_data)
to_save_data = {}
for i, (img_id, ex_data) in tqdm(enumerate(extra_data.items()), total=len(extra_data)):
    in_data = intra_data.get(img_id, None)
    if validate_external:
        if not in_data:
            tmp = ex_data_to_in_data(ex_data, img_infos)
            if len(tmp["relations"]) == 0:
                continue
            to_save_data[img_id] = tmp

    if in_data:
        to_save_data[img_id] = merge_ex_and_in(ex_data, in_data)
to_save_data = list(to_save_data.values())
os.makedirs("Final", exist_ok=True)
if validate_external and validate_internal:
    pickle.dump(to_save_data, open("Final/yes_ext_yes_int.pk", "wb"))
elif validate_internal and not validate_external:
    pickle.dump(to_save_data, open("Final/no_ext_yes_int.pk", "wb"))
elif not validate_internal and validate_external:
    pickle.dump(to_save_data, open("Final/yes_ext_no_int.pk", "wb"))
else:
    pickle.dump(to_save_data, open("Final/no_ext_no_int.pk", "wb"))

# stat relations
print("saved data:", len(to_save_data), sum([len(x["relations"]) for x in to_save_data]) )