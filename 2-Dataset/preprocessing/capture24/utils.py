import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def is_good_quality(w, WINDOW_LEN, WINDOW_SEC, WINDOW_TOL):
    ''' Window quality check '''
    
    # Check null values in selected windows
    if w.isna().any().any(): 
        return False

    # Check the selected window length
    if len(w) != WINDOW_LEN: 
        return False

    # Check the selected windows label variety
    if len(w['annotation'].unique()) > 1: 
        return False

    # Check the validation of selected windows timeframe
    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, 's')
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False

    return True


def resize(X, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """
    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )
    return X

def ftDataset_to_ptDataset(OUTDIR, pid_unique, pid, X):

    os.system(f'mkdir -p {OUTDIR}')

    file_list = []
    for i, p in enumerate(tqdm(pid_unique)):
        idx = np.where(pid == p)
        X_p = X[idx,:,:][0]
        new_X_p = []
        for w_p in X_p:
            w_std_p = np.expand_dims(w_p.std(axis=0), axis=0)
            new_w_p = np.concatenate((w_p, w_std_p), axis=0).transpose()
            new_X_p.append(new_w_p)
        filepath = os.path.join(OUTDIR, f'{p}.npy')
        np.save(filepath, new_X_p)
        file_list.append(filepath)

    filepath = os.path.join(OUTDIR, 'file_list.csv')
    df_files = pd.DataFrame(file_list, columns=['file_list'])
    df_files.to_csv(filepath,index=False)

def create_pid_info(CURRENT,CAP24_FT_ROOT,CAP24_ROOT):
    pid = np.load(CAP24_FT_ROOT+'pid.npy')
    Y = np.load(CAP24_FT_ROOT+'Y.npy')
    metadata = pd.read_csv(CAP24_ROOT+'metadata.csv')

    df = pd.DataFrame({'pid':pid, 'Y':Y})

    pid_label_cnt = pd.DataFrame({'pid':[],'sleep':[], 'sit-stand':[], 'walking':[], 'mixed':[], 'vehicle':[], 'bicycling':[]})

    for pid in np.sort(df['pid'].unique()):
        _ = df.loc[df['pid']==pid].groupby('Y').count().to_dict()['pid']
        _['pid'] = pid
        _df = pd.DataFrame(_, index=[0])
        pid_label_cnt = pd.concat([pid_label_cnt,_df],ignore_index=True)

    result = pd.merge(metadata, pid_label_cnt, on="pid")
    pid_info_path = os.path.join(CURRENT,'pid_info.csv')
    result.to_csv(pid_info_path)

    return pid_info_path

def get_test_pids(CAP24_ROOT,pid_info_path):

    metadata = pd.read_csv(CAP24_ROOT+'metadata.csv')
    result = pd.read_csv(pid_info_path)
    labels = ['sleep',  'sit-stand',  'walking',   'mixed',  'vehicle',  'bicycling']

    p_set1 = metadata.loc[(metadata['age']=='18-29') & (metadata['sex'] == 'F')]['pid']
    p_set2 = metadata.loc[(metadata['age']=='18-29') & (metadata['sex'] == 'M')]['pid']
    p_set3 = metadata.loc[(metadata['age']=='30-37') & (metadata['sex'] == 'F')]['pid']
    p_set4 = metadata.loc[(metadata['age']=='30-37') & (metadata['sex'] == 'M')]['pid']
    p_set5 = metadata.loc[(metadata['age']=='38-52') & (metadata['sex'] == 'F')]['pid']
    p_set6 = metadata.loc[(metadata['age']=='38-52') & (metadata['sex'] == 'M')]['pid']
    p_set7 = metadata.loc[(metadata['age']=='53+') & (metadata['sex'] == 'F')]['pid']
    p_set8 = metadata.loc[(metadata['age']=='53+') & (metadata['sex'] == 'M')]['pid']

    test_pids = []
    for p_set in [p_set1,p_set2,p_set3,p_set4,p_set5,p_set6,p_set7,p_set8,]:
        comb_pids = list(combinations(p_set, 5))

        vs = []
        scores = []
        for pids in comb_pids:
            v = result.loc[result['pid'].isin(list(pids))][labels].sum().values
            vs.append(v)
            score = np.std(v)
            scores.append(score)
            
        idx = np.array(scores).argmin()
        print(comb_pids[idx])
        print(vs[idx])
        test_pids += list(comb_pids[idx])

    sum_test_result = result.loc[result['pid'].isin(test_pids)][labels].sum().sum()
    sum_test_label_result = result.loc[result['pid'].isin(test_pids)][labels].sum()

    sum_train_result = result.loc[~result['pid'].isin(test_pids)][labels].sum().sum()
    sum_train_label_result = result.loc[~result['pid'].isin(test_pids)][labels].sum()

    sum_result = result[labels].sum().sum()
    sum_label_result = result[labels].sum()

    print()
    print(test_pids)
    print(sum_test_result/sum_result)
    print(sum_test_label_result/sum_label_result)

    return test_pids
