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

def target_load(model:str, feats_out=False):
    t_true = None
    t_pred = None
    t_logits = None
    t_feats = None

    for idx, k in enumerate(load_data[model].keys()):

        if k not in ['source', 'source+unknown']:
            true = np.load(load_data[model][k]['true'])
            pred = np.load(load_data[model][k]['pred'])
            logits = np.load(load_data[model][k]['logits'])
            if feats_out:
                feats = np.load(load_data[model][k]['feats'])
            else:
                feats = None

            if not isinstance(t_true, np.ndarray):
                t_true = true
                t_pred = pred
                t_logits = logits
                if feats_out:
                    t_feats = feats
                print(f"append {k} {true.shape}")
                continue

            t_true = np.append(t_true, true, axis=0)
            t_pred = np.append(t_pred, pred, axis=0)
            t_logits = np.append(t_logits, logits, axis=0)
            if feats_out:
                t_feats = np.append(t_feats, feats, axis=0)
            print(f"append {k} {true.shape}")
    
    print(np.unique(t_true, return_counts=True))
    print(np.unique(t_pred, return_counts=True))

    return t_true, t_pred, t_logits, t_feats

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
        t_true, t_pred, t_logits, _ = target_load(m)
        print('true label stats: ',np.unique(t_true,return_counts=True))
        print('pred label stats: ',np.unique(t_pred,return_counts=True))
        print('logits shape: ', t_logits.shape)
        B_CCR, B_FPR, Threshold = utilA.create_balanced_oscr_curve_gpu(t_true,t_pred,t_logits, my_device)
        # print(B_CCR, B_FPR, Threshold)
        print(f'DONE - target oscr {m} model')
        np.save(f'./oscr_result/gpu/t_{m}_CCR.npy',B_CCR)
        np.save(f'./oscr_result/gpu/t_{m}_FPR.npy',B_FPR)
        np.save(f'./oscr_result/gpu/t_{m}_Threshold.npy',Threshold)

if __name__ == "__main__":
    main()
