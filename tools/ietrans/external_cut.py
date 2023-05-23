import json
import pickle
import numpy as np
import os
import torch
ratio = 1.0
print(ratio)
score = json.load(open("External/score.json", "r"))
em = pickle.load(open("External/raw_em_E.pk", "rb"))
if ratio == 1.0:
    thres = score[-1]
elif ratio == 0.0:
    thres = score[0]
else:
    thres = score[int(ratio * len(score))-1]

n = 0
rst = []
for d in em:
    pairs = d['pairs']
    rel_logits = d['rel_logits']
    possible_rels = d['possible_rels']
    rst_pairs = []
    rst_rel_logits = []
    rst_possible_rels = []
    for i, (logit, rels) in enumerate(zip(rel_logits, possible_rels)):
        s = torch.tensor(logit).softmax(0)[-1]
        if s > thres:
            # do not transfer
            # rst_pairs.append(pairs[i])

            continue
        else:
            # transfer
            rst_pairs.append(pairs[i])
            rst_rel_logits.append(logit[0:-1])
            rst_possible_rels.append(rels[:-1])
    d['pairs'] = np.asarray(rst_pairs)
    d['rel_logits'] = rst_rel_logits
    d['possible_rels'] = rst_possible_rels
    if rst_rel_logits == []:
        n += 1
        rst.append(None)
    else:
        rst.append(d)
# pickle.dump(rst, open("em_E.pk"+str(round(ratio, 2)), "wb"))
os.makedirs("External", exist_ok=True)
pickle.dump(rst, open(f"External/External_{ratio}.pk", "wb"))
print(n)