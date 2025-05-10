import os
import numpy as np
import utilAnalysis as utilA
import constants as const
import torch

root = '...'
# model = const.VARIANTS[:7].copy()
model = const.VARIANTS.copy()
model.remove('var-U')
print(model)
dataset = dict(zip(const.DS_LABEL, const.DS_NAME))
dataset_alias = list(dataset.keys())

load_data = {}
for m in model:
    item = {}
    for d_a in dataset_alias:
        _ = {}
        _['name'] = dataset[d_a]
        _['true'] = os.path.join(root, m, _['name'], 'fold0', 'true_Y.npy')
        _['pred'] = os.path.join(root, m, _['name'], 'fold0', 'pred_Y.npy')
        _['logits'] = os.path.join(root, m, _['name'], 'fold0', 'logits.npy')
        _['feats'] = os.path.join(root, m, _['name'], 'fold0', 'feats.npy')
        item[d_a] = _
    load_data[m] = item

def walk_dict(d,depth=0):
    for k,v in sorted(d.items(),key=lambda x: x[0]):
        space = "  "*depth
        if isinstance(v, dict):
            print(f"{space}{k}")
            walk_dict(v,depth+1)
        else:
            print(f"{space}{k} {v}")
            # print ("  ")*depth + "%s %s" % (k, v) 

def source_load(model:str, unknown=True, feats_out=False):

    if unknown:
        s_true = np.load(load_data[model]['source+unknown']['true'])
        s_pred = np.load(load_data[model]['source+unknown']['pred'])
        s_logits = np.load(load_data[model]['source+unknown']['logits'])
        if feats_out:
            s_feats = np.load(load_data[model]['source+unknown']['feats'])
        else:
            s_feats = None
    else:
        s_true = np.load(load_data[model]['source']['true'])
        s_pred = np.load(load_data[model]['source']['pred'])
        s_logits = np.load(load_data[model]['source']['logits'])
        if feats_out:
            s_feats = np.load(load_data[model]['source']['feats'])
        else:
            s_feats = None
    # print(idxs)
    return s_true, s_pred, s_logits, s_feats

def main():

    walk_dict(load_data, depth=0)

    # GPU Setting
    my_device = "cpu"
    if torch.cuda.is_available():
        print(
            'cuda device counts: ',torch.cuda.device_count(), 
            '\ncuda current device: ', torch.cuda.current_device(), 
            '\ncuda device name: ', torch.cuda.get_device_name(0))
        my_device = "cuda:0"
    print(f"GPU using : {my_device}")

    for m in model:
        print(f"oscr for {m}")
        s_true, s_pred, s_logits, _ = source_load(m)
        print('true label stats: ',np.unique(s_true,return_counts=True))
        print('pred label stats: ',np.unique(s_pred,return_counts=True))
        print('logits shape: ', s_logits.shape)
        B_CCR, B_FPR, Threshold = utilA.create_balanced_oscr_curve_gpu(s_true,s_pred,s_logits, my_device)
        # print(B_CCR, B_FPR, Threshold)
        print(f'DONE - source oscr {m} model')
        np.save(f'./oscr_result/gpu/{m}_CCR.npy',B_CCR)
        np.save(f'./oscr_result/gpu/{m}_FPR.npy',B_FPR)
        np.save(f'./oscr_result/gpu/{m}_Threshold.npy',Threshold)

if __name__ == "__main__":
    main()
