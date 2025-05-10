# Source from: https://github.com/OxWearables/ssl-wearables
# modifided by Hyeongkyun Kim(hyeongkyun.kim@uzh.ch) and Orestis Oikonomou(orestis.oikonomou@uzh.ch)
# ---------------------------------------------------------------
# Copyright © 2022, University of Oxford
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_mtssl

import os
import re
import sys
source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(source_dir)
sys.path.append(source_dir)

import hydra
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import collections

from torch.utils.data import DataLoader
from torch.autograd import Variable
from datetime import datetime
from omegaconf import OmegaConf

from sslearning.models.accNet import Resnet
from sslearning.data.data_loader import NormalDataset


##############################################
# etc.
##############################################
def get_class_weights(y):
    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Weight tensor: ")
    print(weights)
    return weights

def folder_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")

def result_path_create(cfg):
    model_root = cfg.model_root
    fold_pattern = re.compile("fold[0-9]")

    # Scan .pt files in the model root
    root_child = [ name for name in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, name)) ]
    fold_list = [ name for name in root_child if fold_pattern.match(name) ]
    model_path = []
    for fold in fold_list:
        fold_path = os.path.join(model_root, fold)
        for file in os.listdir(fold_path):
            if file.endswith(".pt"):
                model_path.append(os.path.join(fold_path,file))

    # Create folders for pred_Y.npy
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    result_path = []
    for model in model_path:
        folder,_ = os.path.split(model)
        folder = folder.replace('_finetuning','_inference')
        folder = folder.replace('_model','_result')
        folder = folder.replace("_result", "_result" + "/" + dt_string)
        folder,fold = os.path.split(folder)
        folder,_ = os.path.split(folder)
        result = os.path.join(folder, cfg.output_prefix, fold)
        folder_exist(result)
        result_path.append(result)
    
    return result_path, model_path

##############################################
# Test Data Loader
##############################################
def setup_test_data(eval_X, eval_Y, groups, dataset_name, cfg):
    test_dataset = NormalDataset(
                                eval_X, eval_Y, pid=groups, name=dataset_name, isLabel=True
                            )
    test_loader = DataLoader(
                                test_dataset,
                                batch_size=cfg.data.batch_size,
                                num_workers=cfg.evaluation.num_workers,
                            )

    return test_loader

##############################################
# MTSSL Model Load
##############################################
def init_model(cfg, my_device):

    output_size = len(cfg.data.cap24_labels)
    if 'unknown' in cfg.data.cap24_labels:
        # Keep the size of output layer as the number of known labels.
        output_size -= 1

    model = Resnet(
        output_size=output_size,
        epoch_len=cfg.dataloader.epoch_len,
        add_classical_feats=cfg.add_classical_feats,
        sample_rate=cfg.data.sample_rate,
        my_device=my_device
    )

    if cfg.multi_gpu:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)

    model.to(my_device, dtype=torch.float)

    return model

##############################################
# Inference Main
##############################################

def inference_mlp(eval_X, eval_Y, cfg, my_device, logger, groups=None):

    # ----------------------------
    # 
    #     Prepare DataLoader``
    #
    # ----------------------------

    # PyTorch defaults to float32
    # channels first: (N,M,3) -> (N,3,M). PyTorch uses channel first format
    eval_X = np.transpose(eval_X.astype("f4") , (0, 2, 1))
    print("X transformed shape:", eval_X.shape)

    # encode ft_Y values into integer
    labels = cfg.data.cap24_labels
    eval_Y = [labels.index(Y) for Y in eval_Y]
    print('Encoded actual true_Y labels into int values with following orders: ', labels)
    print("true_Y unique values (6 is 'unknown'): ", np.unique(eval_Y))
    test_loader = setup_test_data(eval_X, eval_Y, groups, 'cap24_test', cfg)

    for model_path,result_path in zip(cfg.model_path, cfg.result_path):
        # ----------------------------
        # 
        #     Prepare model
        #
        # ----------------------------
        torch_load = torch.load(model_path)
        model = init_model(cfg, my_device)
        if 'model_state_dict' in torch_load:
            model.load_state_dict(torch_load['model_state_dict'])
        else:
            model.load_state_dict(torch_load)
        model.eval()

        # ----------------------------
        # 
        #     Inference Y label
        #
        # ----------------------------
        print(f"\n\nInference Start!\nModel : {model_path}")
        logits_list = []
        final_feats_list = []
        pred_list = []
        true_list = []
        pid_list = []
        for i, (my_X, my_Y, my_PID) in enumerate(test_loader):
            with torch.no_grad():
                my_X, my_Y = Variable(my_X), Variable(my_Y)
                my_X = my_X.to(my_device, dtype=torch.float)

                true_y = my_Y.to(my_device, dtype=torch.long)
                logits, final_feats = model(my_X)
                if cfg.save_logits:
                    logits_list += logits.tolist()
                if cfg.save_final_feats:
                    final_feats_list += final_feats.tolist()

                pred_y = torch.argmax(logits, dim=1)
                pred_list.append(pred_y.cpu())

                true_list.append(true_y.cpu())
                pid_list.extend(my_PID)

        pred_list = torch.cat(pred_list)
        true_list = torch.cat(true_list)
        pred_Y = torch.flatten(pred_list).numpy()
        true_Y = torch.flatten(true_list).numpy()
        pids = np.array(pid_list)

        # decode pred_Y and true_Y values into dataset labels
        pred_Y = np.array([labels[int(Y)] for Y in pred_Y])
        true_Y = np.array([labels[int(Y)] for Y in true_Y])
        print('Decoded int pred_Y values into actual Y labels : ', np.unique(pred_Y))
        print('Decoded int true_Y values into actual Y labels : ', np.unique(true_Y))

        # ----------------------------
        # 
        #     Save Predicted Y label
        #
        # ----------------------------
        print(f"\nTrue Y Statics")
        print(pd.Series(true_Y).value_counts())
        print(f"\nPredicted Y Statics")
        print(pd.Series(pred_Y).value_counts())

        # To save large file, Path create on Euler SCRATCH.
        large_result_path_root = os.getenv("SCRATCH") + "/mp-dataset/__cache__/"
        result_path_part = os.path.normpath(result_path).split(os.path.sep)[-6:]
        large_result_path = os.path.join(large_result_path_root, *result_path_part)
        folder_exist(large_result_path)

        if cfg.save_logits:
            logits_arr = np.array(logits_list)
            logits_path = os.path.join(large_result_path,'logits.npy')

            print(f"\nSave logits!\nResult: {logits_path}")
            np.save(logits_path, logits_arr)
        
        if cfg.save_final_feats:
            final_feats_arr = np.array(final_feats_list)
            final_feats_path = os.path.join(large_result_path,'feats.npy')
            
            print(f"\nSave final features!\nResult: {final_feats_path}")
            np.save(final_feats_path, final_feats_arr)


        ref_true_y_path = os.path.join(large_result_path,'true_Y.npy')
        print(f"\nSave true labels!\nResult: {ref_true_y_path}")
        np.save(ref_true_y_path, true_Y)

        ref_pred_y_path = os.path.join(large_result_path,'pred_Y.npy')
        print(f"\nSave predicted labels!\nResult: {ref_pred_y_path}")
        np.save(ref_pred_y_path, pred_Y)

        ref_pid_y_path = os.path.join(large_result_path,'pid.npy')
        print(f"\nSave PIDs!\nResult: {ref_pid_y_path}")
        np.save(ref_pid_y_path, pids)


##############################################
# Main
##############################################
@hydra.main(config_path=os.path.join(source_dir, '_conf'), config_name="config_inf", version_base="1.1")
def main(cfg):

    # ----------------------------
    # 
    #       Setting
    #
    # ----------------------------
    # Set output file path and GPU.
    # (output_prefix)_(dataset_name)_(%d%m%Y_%H%M%S).log
    # (output_prefix)_(dataset_name)_(%d%m%Y_%H%M%S).pt
    # ----------------------------
    print(
        "\n\n----------------------------\
        \n\n\t Setting\
        \n\n----------------------------\n\n"
    )

    # File name and path setting.
    cfg.result_path, cfg.model_path = result_path_create(cfg)
    output_prefix = cfg.output_prefix

    # Logger setting
    output_root = os.path.dirname(cfg.result_path[0])
    logger = logging.getLogger(cfg.evaluation.evaluation_name)
    logger.setLevel(logging.INFO)
    log_dir = os.path.join(output_root,output_prefix + ".log",)
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info(str(OmegaConf.to_yaml(cfg)))

    # For reproducibility, fix the seed.
    np.random.seed(42)
    torch.manual_seed(42)

    # GPU Setting
    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:0"
    elif cfg.multi_gpu is True:
        my_device = "cuda:0"  # use the first GPU as master
    else:
        my_device = "cpu"

    print(f"Logging file : {log_dir}")
    print(f"Fintuned model : {cfg.model_path}")
    print(f"Evaluation Report : {cfg.result_path}")
    if torch.cuda.is_available():
        print(
            'cuda device counts: ',torch.cuda.device_count(), 
            '\ncuda current device: ', torch.cuda.current_device(), 
            '\ncuda device name: ', torch.cuda.get_device_name(0))
    print(f"GPU using : {my_device}")


    # ----------------------------
    # 
    #       DataLoading
    #
    # ----------------------------
    # Load dataset for finetuning and evaluation.
    # finetuning dataset would be downsized on purpose.
    print(
        "\n\n----------------------------\
        \n\n\t DataLoading\
        \n\n----------------------------\n\n"
    )

    sample_rate = cfg.data.sample_rate

    # Load evaluation dataset
    X = np.load(cfg.data.X_path)
    Y = np.load(cfg.data.Y_path)
    P = np.load(cfg.data.PID_path)

    eval_pids = cfg.data.pid_filter
    if eval_pids is not None:
        eval_idx = np.where([True if p in eval_pids else False for p in P])
        eval_X, eval_Y, eval_P = X[eval_idx], Y[eval_idx], P[eval_idx]
        print(f"Particular samples are designated for evaluation in this dataset.")
        print(f"Select evaluation samples: {len(eval_idx[0])} ea")
    else:
        eval_X, eval_Y, eval_P = X, Y, P

    # Add unknown unknown label into the evaluation dataset
    if cfg.data.add_unknown:

        unkn_unkn_X = np.load(cfg.data.unknown_unknown_X_path)
        unkn_unkn_Y = np.load(cfg.data.unknown_unknown_Y_path)
        unkn_unkn_PID = np.load(cfg.data.unknown_unknown_PID_path)
        eval_X = np.concatenate([eval_X, unkn_unkn_X])
        eval_Y = np.concatenate([eval_Y, unkn_unkn_Y])
        eval_P = np.concatenate([eval_P, unkn_unkn_PID])
        
        print("Unknown Unknown label added!")
        print(f"unkn_unkn_X: {unkn_unkn_X.shape}")
        print(f"unkn_unkn_Y: {unkn_unkn_Y.shape}")
        print(f"unkn_unkn_PID: {unkn_unkn_PID.shape}")
        print(np.unique(unkn_unkn_PID, return_counts=True))

        # Add 'unknown' label into the cap24 label definition
        ext_cap24_labels = cfg.data.cap24_labels 
        ext_cap24_labels.append('unknown')
        cfg.data.cap24_labels = ext_cap24_labels
        print(f"New Cap24 Label is set...\n{cfg.data.cap24_labels}")

    print("DataLoading")
    print(f"Inference Dataset: {cfg.data.dataset_name}")
    print(f"\t X shape: {eval_X.shape}")
    print(f"\t Y shape: {eval_Y.shape}")
    print(f"\t PID shape: {eval_P.shape}")

    unique, counts = np.unique(eval_Y, return_counts=True)
    print(f"Y labels :\t{unique}\ncounts :\t{counts}")

    unknown_idx = np.where(eval_Y == 'unknown')[0]
    unknown_cnt = len(unknown_idx)

    if unknown_cnt > 0:
        if cfg.data.inf_use_unknown:
            ext_cap24_labels = cfg.data.cap24_labels
            if 'unknown' not in ext_cap24_labels:
                ext_cap24_labels.append('unknown')
                cfg.data.cap24_labels = ext_cap24_labels
                print(f"New Cap24 Label is set...\n{cfg.data.cap24_labels}")
        else:
            print(f"Filter out unknown labels : {unknown_cnt}ea")
            eval_X = np.delete(eval_X, unknown_idx, axis=0)
            eval_Y = np.delete(eval_Y, unknown_idx, axis=0)
            eval_P = np.delete(eval_P, unknown_idx, axis=0)
            unique, counts = np.unique(eval_Y, return_counts=True)
            print(f"Y labels :\t{unique}\ncounts :\t{counts}")
    
    # ----------------------------
    # 
    #       MTSSL | Inference
    #
    # ----------------------------
    print(
        "\n\n----------------------------\
        \n\n\t MTSSL | Inference\
        \n\n----------------------------\n\n"
    )
    inference_mlp(eval_X, eval_Y, cfg, my_device, logger, groups=eval_P)

if __name__ == "__main__":
    main()
