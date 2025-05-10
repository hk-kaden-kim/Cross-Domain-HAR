import os
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = "..." 

NEGATIVE = ["gotov", "pamap2", "wisdm", "harvardleo", "householdhu", "realworld"]
UNKNOWN = ["paal", "commuting"]
print()

#######################################################
# Load Negative samples
#######################################################
print(f"Create negative samples: {NEGATIVE}")
print()
for idx, neg in enumerate(NEGATIVE):
    file_path = f"{ROOT}/finetune_{neg}_30hz_w10"
    X = np.load(os.path.join(file_path, "relabel", "Willetts2018", "X.npy"))
    Y = np.load(os.path.join(file_path, "relabel", "Willetts2018", "Y.npy"))

    # Filter out only "unknown"
    unkn_mask = [True if y=="unknown" else False for y in Y]
    X = X[unkn_mask]
    Y = Y[unkn_mask]
    PID = np.array([f"unknown_{neg}"]*len(Y))

    if idx == 0:
        neg_X = X
        neg_Y = Y
        neg_PID = PID
    else:
        neg_X = np.concatenate((neg_X,X), axis=0)
        neg_Y = np.concatenate((neg_Y,Y), axis=0)
        neg_PID = np.concatenate((neg_PID,PID), axis=0)

print("Statistics for all unknown samples from additional 6 dataset...")
print(np.unique(neg_Y, return_counts=True))
print(np.unique(neg_PID, return_counts=True))
print()

#######################################################
# Split Negative samples into fine-tuning (8) and testing (2)
#######################################################
idx = np.arange(len(neg_Y))
ft_idx, test_idx, _, _ = train_test_split(idx, [0]*len(idx), test_size=0.2, random_state=42)

ft_neg_X = neg_X[ft_idx]
ft_neg_Y = neg_Y[ft_idx]
ft_neg_PID = neg_PID[ft_idx]

test_neg_X = neg_X[test_idx]
test_neg_Y = neg_Y[test_idx]
test_neg_PID = neg_PID[test_idx]

print("Statistics for all negative samples in finetuning dataset...")
print(np.unique(ft_neg_Y, return_counts=True))
print(np.unique(ft_neg_PID, return_counts=True))
print()

print("Statistics for all negative samples in testing dataset...")
print(np.unique(test_neg_Y, return_counts=True))
print(np.unique(test_neg_PID, return_counts=True))
print()

#######################################################
# Load (UNKNOWN) UNKNOWN samples
#######################################################
print(f"Create unknown unknown samples: {UNKNOWN}")
print()
for idx, unkn_dataset in enumerate(UNKNOWN):
    file_path = f"{ROOT}/finetune_{unkn_dataset}_30hz_w10"
    X = np.load(os.path.join(file_path, "relabel", "Willetts2018", "X.npy"))
    Y = np.load(os.path.join(file_path, "relabel", "Willetts2018", "Y.npy"))

    # Filter out only "unknown"
    unkn_mask = [True if y=="unknown" else False for y in Y]
    X = X[unkn_mask]
    Y = Y[unkn_mask]
    PID = np.array([f"unknown_{unkn_dataset}"]*len(Y))

    if idx == 0:
        test_unkn_X = X
        test_unkn_Y = Y
        test_unkn_PID = PID
    else:
        test_unkn_X = np.concatenate((test_unkn_X,X), axis=0)
        test_unkn_Y = np.concatenate((test_unkn_Y,Y), axis=0)
        test_unkn_PID = np.concatenate((test_unkn_PID,PID), axis=0)

print("Statistics for all unknown unknown samples in testing dataset...")
print(np.unique(test_unkn_Y, return_counts=True))
print(np.unique(test_unkn_PID, return_counts=True))
print()


#######################################################
# Merge Test Negative samples and (UNKNOWN) UNKNOWN samples
#######################################################
test_unkn_all_X = np.concatenate((test_neg_X, test_unkn_X), axis=0)
test_unkn_all_Y = np.concatenate((test_neg_Y, test_unkn_Y), axis=0)
test_unkn_all_PID = np.concatenate((test_neg_PID, test_unkn_PID), axis=0)



#######################################################
# Save
#######################################################
os.system(f"mkdir -p {ROOT}/finetune_negative_30hz_w10/relabel/Willetts2018")
np.save(os.path.join(f"{ROOT}/finetune_negative_30hz_w10/relabel/Willetts2018","X.npy"),ft_neg_X)
np.save(os.path.join(f"{ROOT}/finetune_negative_30hz_w10/relabel/Willetts2018","Y.npy"),ft_neg_Y)
np.save(os.path.join(f"{ROOT}/finetune_negative_30hz_w10/relabel/Willetts2018","pid.npy"),ft_neg_PID)
print(f"Done! Check here -> {ROOT}/finetune_negative_30hz_w10/relabel/Willetts2018")

os.system(f"mkdir -p {ROOT}/test_negative_30hz_w10/relabel/Willetts2018")
np.save(os.path.join(f"{ROOT}/test_negative_30hz_w10/relabel/Willetts2018","X.npy"),test_neg_X)
np.save(os.path.join(f"{ROOT}/test_negative_30hz_w10/relabel/Willetts2018","Y.npy"),test_neg_Y)
np.save(os.path.join(f"{ROOT}/test_negative_30hz_w10/relabel/Willetts2018","pid.npy"),test_neg_PID)
print(f"Done! Check here -> {ROOT}/test_negative_30hz_w10/relabel/Willetts2018")

os.system(f"mkdir -p {ROOT}/test_unknown_30hz_w10/relabel/Willetts2018")
np.save(os.path.join(f"{ROOT}/test_unknown_30hz_w10/relabel/Willetts2018","X.npy"),test_unkn_X)
np.save(os.path.join(f"{ROOT}/test_unknown_30hz_w10/relabel/Willetts2018","Y.npy"),test_unkn_Y)
np.save(os.path.join(f"{ROOT}/test_unknown_30hz_w10/relabel/Willetts2018","pid.npy"),test_unkn_PID)
print(f"Done! Check here -> {ROOT}/test_unknown_30hz_w10/relabel/Willetts2018")

os.system(f"mkdir -p {ROOT}/test_unknown_all_30hz_w10/relabel/Willetts2018")
np.save(os.path.join(f"{ROOT}/test_unknown_all_30hz_w10/relabel/Willetts2018","X.npy"),test_unkn_all_X)
np.save(os.path.join(f"{ROOT}/test_unknown_all_30hz_w10/relabel/Willetts2018","Y.npy"),test_unkn_all_Y)
np.save(os.path.join(f"{ROOT}/test_unknown_all_30hz_w10/relabel/Willetts2018","pid.npy"),test_unkn_all_PID)
print(f"Done! Check here -> {ROOT}/test_unknown_all_30hz_w10/relabel/Willetts2018")