import os
import numpy as np
import pandas as pd
from itertools import combinations

from utils import create_pid_info, get_test_pids

CURRENT = os. getcwd()
SCRATCH = os.environ['SCRATCH']

CAP24_FT_ROOT = SCRATCH + '/mp-dataset/finetune_capture24_30hz_w10/Willetts2018/'
CAP24_ROOT = SCRATCH + '/mp-dataset/capture24/capture24/'

# pid_info_path = create_pid_info(CURRENT,CAP24_FT_ROOT,CAP24_ROOT)
# test_pids = get_test_pids(CAP24_ROOT,pid_info_path)

test_pids = ['P023', 'P094', 'P106', 'P112', 'P136',
             'P020', 'P088', 'P123', 'P129', 'P139',
             'P002', 'P003', 'P059', 'P147', 'P149', 
             'P021', 'P036', 'P041', 'P082', 'P135', 
             'P018', 'P052', 'P095', 'P118', 'P121', 
             'P045', 'P055', 'P102', 'P108', 'P144', 
             'P009', 'P081', 'P117', 'P122', 'P140', 
             'P039', 'P069', 'P076', 'P119', 'P142']

pid_info_path = './pid_info.csv'
pid_info = pd.read_csv(pid_info_path)
labels = ['sleep',  'sit-stand',  'walking',   'mixed',  'vehicle',  'bicycling']

print('Capture24 finetuning/evaluation dataset info')
df = pid_info.loc[pid_info['pid'].isin(test_pids)]

sum_test_result = pid_info.loc[pid_info['pid'].isin(test_pids)][labels].sum().sum()
sum_test_result_bylabel = pid_info.loc[pid_info['pid'].isin(test_pids)][labels].sum()

sum_train_result = pid_info.loc[~pid_info['pid'].isin(test_pids)][labels].sum().sum()
sum_train_result_bylabel = pid_info.loc[~pid_info['pid'].isin(test_pids)][labels].sum()

sum_result = pid_info[labels].sum().sum()
sum_result_bylabel = pid_info[labels].sum()

print()
print("----------------Finetuning----------------")
print()
print(sum_train_result)
print(sum_train_result/sum_result)
print()
print(sum_train_result_bylabel)
print(sum_train_result_bylabel/sum_result_bylabel)
print()
print("----------------Evaluation----------------")
print(test_pids)
print()
print(sum_test_result)
print(sum_test_result/sum_result)
print()
print(sum_test_result_bylabel)
print(sum_test_result_bylabel/sum_result_bylabel)