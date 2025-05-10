# Source from: https://github.com/OxWearables/ssl-wearables
# modifided by Hyeongkyun Kim(hyeongkyun.kim@uzh.ch) and Orestis Oikonomou(orestis.oikonomou@uzh.ch)
# ---------------------------------------------------------------
# Copyright © 2022, University of Oxford
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_mtssl

import os
import sys
source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(source_dir)
sys.path.append(source_dir)

import hydra
import copy
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import collections

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from datetime import datetime
from omegaconf import OmegaConf
from sklearn.model_selection import GroupShuffleSplit

from sslearning.models.accNet import Resnet
from sslearning.data.datautils import RandomSwitchAxis, RotationAxis, AmplitudeScale
from sslearning.data.data_loader import NormalDataset
from sslearning.pytorchtools import EarlyStopping
import torch.nn.functional as F



class ObjectosphereLoss(torch.nn.Module):
    def __init__(self, xi, lambda_scale, num_classes, labels, weights):
        super(ObjectosphereLoss, self).__init__()
        self.xi = xi
        self.lambda_scale = lambda_scale
        self.num_classes = num_classes
        self.labels = labels
        self.weights = weights
        self.known_num_classes = sum([True if l != "unknown" else False for l in labels])
        
        print()
        print(f"ObjectosphereLoss Settings!")
        print(f"hyperparameter:\nxi = {self.xi}\tlambda_scale = {self.lambda_scale}")
        print(f"label and weights:\nlabel {self.num_classes}ea = {self.labels}\nweights = {self.weights}")
        print()

    def forward(self, logits, true_y, is_unknown, deep_features):

        # Compute softmax scores
        softmax_scores = F.softmax(logits, dim=1)

        JR_batch = []

        # Calculate L2 norm at the deep feature space.
        feature_magnitude = torch.norm(deep_features, dim=1)
        weights_sum = 0
        for i,y in enumerate(true_y):
            if is_unknown[i]:
                # If the true_y is corresponding to 'unknown' value
                JE_unknown_loss = -(1.0 / self.known_num_classes) * torch.sum(torch.log(softmax_scores[i]+ 1e-10))
                unknown_magnitude_penalty = feature_magnitude[i]**2
                JR = JE_unknown_loss + self.lambda_scale*unknown_magnitude_penalty
            else:
                # If the true_y is corresponding to 'known' value
                JE_known_loss = -torch.log(softmax_scores[i,y]+ 1e-10)
                # For unknown samples
                known_magnitude_penalty = torch.clamp(self.xi - feature_magnitude[i],min=0)**2
                JR = JE_known_loss + self.lambda_scale*known_magnitude_penalty
            
            # Weighted JR
            JR = self.weights[y]*JR 
            weights_sum += self.weights[y]
            JR_batch.append(JR) # the weighted mean of the output is taken

        loss = sum(JR_batch)/weights_sum # Weighted mean of total batch loss
        
        return loss




##############################################
# etc.
##############################################
def get_class_weights(y,cfg):
    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(cfg.data.cap24_labels)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)

    # If finetuning dataset has unknown labels, consider 'known and unknown weight' additionally
    if 'unknown' in cfg.data.cap24_labels:
        label = cfg.data.cap24_labels
        unkn_cnt = counter[label.index("unknown")]
        kn_cnt = num_samples - unkn_cnt

        unkn_weight = 1/(unkn_cnt/num_samples)
        kn_weight = 1/(kn_cnt/num_samples)
        kn_unkn_weight = [kn_weight]*len(counter)
        kn_unkn_weight[label.index("unknown")] = unkn_weight

        weights = list(np.array(weights) * np.array(kn_unkn_weight))

    print(f"Weights of each label (in order) : {weights}")    
    return weights

def folder_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")

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

def load_weights(weight_path, model, my_device):
    # only need to change weights name when
    # the model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    # distributed pretraining can be inferred from the keys' module. prefix
    head = next(iter(pretrained_dict_v2)).split('.')[0]  # get head of first key
    if head == 'module':
        # remove module. prefix from dict keys
        pretrained_dict_v2 = {k.partition('module.')[2]: pretrained_dict_v2[k] for k in pretrained_dict_v2.keys()}

    if hasattr(model, 'module'):
        model_dict = model.module.state_dict()
        multi_gpu_ft = True
    else:
        model_dict = model.state_dict()
        multi_gpu_ft = False

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    if multi_gpu_ft:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))

def setup_model(cfg, my_device):
    model = init_model(cfg, my_device)

    load_weights(
        cfg.evaluation.flip_net_path,
        model,
        my_device
    )

    return model


##############################################
# Data Loader
##############################################
def train_val_split(X, Y, group, val_size=0.125):
    num_split = 1
    folds = GroupShuffleSplit(
        num_split, test_size=val_size, random_state=41
    ).split(X, Y, groups=group)
    train_idx, val_idx = next(folds)
    return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]

def setup_data(ft_X, ft_Y, ft_P, my_transform, cfg):

    # Split data into train(7/8) and valid(1/8)
    X_train, X_val, Y_train, Y_val = train_val_split(
        ft_X, ft_Y, ft_P
    )

    # Define Dataset for each train and validation set.
    train_dataset = NormalDataset(
        X_train, Y_train, name="train", isLabel=True, transform=my_transform
    )
    val_dataset = NormalDataset(X_val, Y_val, name="val", isLabel=True)

    # Define Dataloader for each train and validatin set.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.evaluation.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )

    weights = get_class_weights(Y_train,cfg)
    
    return train_loader, val_loader, weights


##############################################
# Finetuning
##############################################
def validate_model(model, data_loader, my_device, loss_fn, cfg):
    model.eval()
    losses = []
    acces = []
    labels = cfg.data.cap24_labels
    for i, (my_X, my_Y) in enumerate(data_loader):
        with torch.no_grad():
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            true_y = my_Y.to(my_device, dtype=torch.long)

            logits, deep_features = model(my_X)  # Get both logits and deep features

            # Compute the Objectosphere loss
            if cfg.data.add_unknown:
                is_unknown = (true_y == labels.index('unknown')) 
                loss = loss_fn(logits, true_y, is_unknown, deep_features)
            else:
                loss = loss_fn(logits, true_y)
            
            pred_y = torch.argmax(logits, dim=1)  # This is for monitoring purposes

            test_acc = torch.sum(pred_y == true_y)
            test_acc = test_acc / (list(pred_y.size())[0])

            losses.append(loss.cpu().detach().numpy())
            acces.append(test_acc.cpu().detach().numpy())

    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces)

def finetune_mlp(ft_X, ft_Y, cfg, my_device, logger, groups=None):

    # Define data transformations.
    my_transform = None

    if cfg.data.augmentation == 0:
        my_transform = None
    elif cfg.data.augmentation == 1:
        my_transform = transforms.Compose([RandomSwitchAxis()])
    elif cfg.data.augmentation == 2:
        my_transform = transforms.Compose([RotationAxis()])
    elif cfg.data.augmentation == 3:
        my_transform = transforms.Compose([AmplitudeScale()])
    elif cfg.data.augmentation == 12:
        my_transform = transforms.Compose([RandomSwitchAxis(),RotationAxis()])
    elif cfg.data.augmentation == 13:
        my_transform = transforms.Compose([RandomSwitchAxis(),AmplitudeScale()])
    elif cfg.data.augmentation == 23:
        my_transform = transforms.Compose([RotationAxis(),AmplitudeScale()])
    elif cfg.data.augmentation == 123:
        my_transform = transforms.Compose([RandomSwitchAxis(),RotationAxis(),AmplitudeScale()])
    else:
        assert False, f"Augmentation setting is unknown. got: {cfg.data.augmentation}" 

    print(f"Data Augmentation with...\n\t{my_transform}")

    for fold in range(len(ft_Y)):
        
        # ----------------------------
        # 
        #     Prepare model
        #
        # ----------------------------
        print(f'model: {cfg.evaluation.flip_net_path}')
        model = setup_model(cfg, my_device)
        if cfg.is_verbose and fold == 0: print(model)

        # ----------------------------
        # 
        #     Prepare DataLoader
        #
        # ----------------------------
        print(f'\n\n Fold {fold} | Finetuning | {cfg.data.dataset_name}')
        ft_X_fold = ft_X[fold]
        ft_Y_fold = ft_Y[fold]
        ft_P_fold = groups[fold]

        # PyTorch defaults to float32
        # channels first: (N,M,3) -> (N,3,M). PyTorch uses channel first format
        ft_X_fold = np.transpose(ft_X_fold.astype("f4") , (0, 2, 1))
        print("X transformed shape:", ft_X_fold.shape)

        # Encode y labels into integer
        labels = cfg.data.cap24_labels
        ft_Y_fold = np.array([labels.index(Y) for Y in ft_Y_fold])
        print('Encoded Y labels (ordered) : ', labels)

        train_loader, val_loader, weights = setup_data(ft_X_fold, ft_Y_fold, ft_P_fold, my_transform, cfg)

        # ----------------------------
        # 
        #       Finetuning
        #
        # ----------------------------
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.evaluation.learning_rate, amsgrad=True
        )

        if cfg.data.add_unknown:
            # Use Objectosphere Loss for the loss function
            xi = cfg.known_min_mag
            lambda_scale = cfg.obj_spr_lambda  # we might need to tune this
            num_classes = len(cfg.data.cap24_labels)
            loss_fn = ObjectosphereLoss(xi, lambda_scale, num_classes, labels, weights=weights)
            lossfn_history = {'avg_known_feat_mag':[], 'xi':xi}
        else:
            # Use Cross Entropy Loss for the loss function
            weights = torch.FloatTensor(weights).to(my_device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            lossfn_history = {}

        path_by_fold = cfg.model_path
        path_head, path_tail = os.path.split(path_by_fold)
        model_output_folder = os.path.join(path_head, f'fold{fold}')
        model_output_path = os.path.join(model_output_folder, path_tail)
        folder_exist(model_output_folder)

        early_stopping = EarlyStopping(
            patience=cfg.evaluation.patience, path=model_output_path, verbose=True
        )

        finetune_history = {'train_loss':[],'valid_loss':[]}
        for epoch in range(cfg.evaluation.num_epoch):
            model.train()

            train_losses = []
            train_acces = []
            val_losses = []
            for i, (my_X, my_Y) in enumerate(train_loader):
                my_X, my_Y = Variable(my_X), Variable(my_Y)
                my_X = my_X.to(my_device, dtype=torch.float)
                true_y = my_Y.to(my_device, dtype=torch.long)

                logits, deep_features = model(my_X)  # Get both logits and deep features
                
                # Compute the Objectosphere loss
                if cfg.data.add_unknown:
                    is_unknown = (true_y == labels.index('unknown')) # Check unknown label in a batch and create the mask
                    loss = loss_fn(logits, true_y, is_unknown, deep_features)
                    known_magnitudes = torch.norm(deep_features[~is_unknown], dim=1)
                else:
                    loss = loss_fn(logits, true_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pred_y = torch.argmax(logits, dim=1)  # This is for monitoring purposes

                train_acc = torch.sum(pred_y == true_y)
                train_acc = train_acc / (pred_y.size()[0])
                train_losses.append(loss.cpu().detach().numpy())
                train_acces.append(train_acc.cpu().detach().numpy())
            
            batch_train_losses = np.mean(train_losses)

            val_loss, val_acc = validate_model(
                model, val_loader, my_device, loss_fn, cfg
            )

            finetune_history['train_loss'].append(batch_train_losses)
            finetune_history['valid_loss'].append(val_loss)

            if cfg.data.add_unknown:
                # To monitor ObjectosphereLoss, track the avg. of magnitude
                # on Feature space only for known label.
                # known_magnitudes = torch.norm(deep_features[~is_unknown], dim=1)
                avg_known_magnitude = torch.mean(known_magnitudes).item()
                lossfn_history['avg_known_feat_mag'].append(avg_known_magnitude)
                # Check Early Stopping condition
                early_stopping(val_loss, model, finetune_history, lossfn_history)
            else:
                # Check Early Stopping condition
                early_stopping(val_loss, model, finetune_history)
            
            epoch_len = len(str(cfg.evaluation.num_epoch))
            print_msg = (
                f"[{epoch:>{epoch_len}}/{cfg.evaluation.num_epoch:>{epoch_len}}] "
                + f"train_loss: {batch_train_losses:.5f} "
                + f"valid_loss: {val_loss:.5f}"
            )
            print(print_msg)

            if early_stopping.early_stop:
                print("Early stopping")
                break

##############################################
# Main
##############################################
@hydra.main(config_path=os.path.join(source_dir, '_conf'), config_name="config_ft", version_base="1.1")
def main(cfg):

    # ----------------------------
    # 
    #       Setting
    #
    # ----------------------------
    # Set output file path and GPU.
    # ./_model/(timestamp)/(output_prefix)_(dataset_name).log
    # ./_model/(timestamp)/(fold)/(output_prefix)_(dataset_name).pt
    # ----------------------------
    print(
        "\n\n----------------------------\
        \n\n\t Setting\
        \n\n----------------------------\n\n"
    )

    # File name and path setting.
    logger = logging.getLogger(cfg.evaluation.evaluation_name)
    logger.setLevel(logging.INFO)
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    output_prefix = cfg.output_prefix + "_" + cfg.data.dataset_name
    output_root = os.path.join(hydra.utils.get_original_cwd(), "_model", dt_string, cfg.output_prefix)
    folder_exist(output_root)

    log_dir = os.path.join(output_root,output_prefix + ".log",)
    cfg.model_path = os.path.join(output_root,output_prefix + ".pt")
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

    # Load finetuning dataset
    if cfg.data.dataset_name == 'capture24':
        X = np.load(cfg.data.X_path)
        Y = np.load(cfg.data.Y_path)
        P = np.load(cfg.data.PID_path)

        eval_pids = cfg.data.pid_filter
        eval_idx = np.where([True if p in eval_pids else False for p in P])
        
        ft_X = np.delete(X, eval_idx, axis=0)
        ft_Y = np.delete(Y, eval_idx,)
        ft_P = np.delete(P, eval_idx,)

        # load and append the unknwon data
        if cfg.data.add_unknown:

            kn_unkn_X = np.load(cfg.data.known_unknown_X_path)
            kn_unkn_Y = np.load(cfg.data.known_unknown_Y_path)
            kn_unkn_PID = np.load(cfg.data.known_unknown_PID_path)
            ft_X = np.concatenate([ft_X, kn_unkn_X])
            ft_Y = np.concatenate([ft_Y, kn_unkn_Y])
            ft_P = np.concatenate([ft_P, kn_unkn_PID])
            
            print("Known Unknown label added!")
            print(f"kn_unkn_X: {kn_unkn_X.shape}")
            print(f"kn_unkn_Y: {kn_unkn_Y.shape}")
            print(f"kn_unkn_PID: {kn_unkn_PID.shape}")
            print(np.unique(kn_unkn_PID, return_counts=True))

            # Add 'unknown' label into the cap24 label definition
            ext_cap24_labels = cfg.data.cap24_labels 
            ext_cap24_labels.append('unknown')
            cfg.data.cap24_labels = ext_cap24_labels
            print(f"New Cap24 Label is set...\n{cfg.data.cap24_labels}")

    else:
        ft_X = np.load(cfg.data.X_path)
        ft_Y = np.load(cfg.data.Y_path)
        ft_P = np.load(cfg.data.PID_path)

    print("DataLoading")
    print(f"Finetuning Dataset: {cfg.data.dataset_name}")
    print(f"\t X shape: {ft_X.shape}") # T x ( Sample Rate*Windows(sec) ) x 3,
    print(f"\t Y shape: {ft_Y.shape}") # T,
    print(pd.Series(ft_Y).value_counts())

    ft_X, ft_Y, ft_P = [ft_X], [ft_Y], [ft_P]
    

    # ----------------------------
    # 
    #       MTSSL | Finetuning
    #
    # ----------------------------
    print(
        "\n\n----------------------------\
        \n\n\t MTSSL | Finetuning\
        \n\n----------------------------\n\n"
    )

    finetune_mlp(ft_X, ft_Y, cfg, my_device, logger, groups=ft_P)

if __name__ == "__main__":
    main()
