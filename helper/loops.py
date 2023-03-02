from __future__ import print_function, division
from operator import truediv
import os
import sys
from time import time
import random
from datetime import timedelta
from scipy.special import softmax
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .utils import save_model, AverageMeter, accuracy, bcolors, prob_loss
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from helper.utils import set_trainable
import itertools



def validate(val_loader, model, criterion, device):
    """validation"""
    # validate the model
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    y_true = []
    y_pred = []
    # validation loop
    with torch.no_grad():
        for batch_id, batch_data in enumerate(val_loader):
            for k in batch_data.keys():
                batch_data[k] = batch_data[k].to(device=device, dtype=torch.float32) if k != "labels" else batch_data[k].to(
                    device=device, dtype=torch.long)

            batch_target = batch_data["labels"]
            if len(batch_data.keys()) == 2:
                batch_input = batch_data["img"]
            else:
                raise ValueError("Have not implemeted yet ...")

            batch_pred = model(batch_input)
            loss = criterion(batch_pred, batch_target)
            acc1 = accuracy(batch_pred, batch_target)

            y_true = np.concatenate((y_true, batch_target.cpu().detach().numpy(
            )), axis=0) if len(y_true) else batch_target.cpu().detach().numpy()
            y_pred = np.concatenate((y_pred, batch_pred.cpu().detach().numpy()), axis=0) if len(
                y_pred) else batch_pred.cpu().detach().numpy()

            val_loss.update(loss.item(), batch_target.size(0))
            val_acc.update(acc1[0], batch_target.size(0))
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write(f"Val: {batch_id} / {len(val_loader)} ")

        val_auroc = roc_auc_score(
            y_true=y_true, y_score=softmax(y_pred, axis=1)[:, 1])
        val_precision = precision_score(
            y_true=y_true, y_pred=np.argmax(y_pred, axis=1))
        val_recall = recall_score(
            y_true=y_true, y_pred=np.argmax(y_pred, axis=1))
        val_f1 = f1_score(y_true=y_true, y_pred=np.argmax(y_pred, axis=1))
    return val_loss.avg, val_acc.avg, val_auroc, val_precision, val_recall, val_f1


def test(test_loader, model, criterion, device):
    """validation"""
    # validate the model
    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    y_true = []
    y_pred = []
    penultimates = []
    # validation loop
    with torch.no_grad():
        for batch_data in tqdm(test_loader, ncols=50):
            for k in batch_data.keys():
                batch_data[k] = batch_data[k].to(device=device, dtype=torch.float32) if k != "labels" \
                    else batch_data[k].to(device=device, dtype=torch.long)

            batch_target = batch_data["labels"]
            if len(batch_data.keys()) == 2:
                batch_input = batch_data["img"]
            else:
                raise ValueError("Have not implemeted yet ...")

            # batch_pred = model(batch_input)
            ft1, ft2, ft3, ft4, penultimate_pred, batch_pred = model(batch_input, True)
            loss = criterion(batch_pred, batch_target)
            acc1 = accuracy(batch_pred, batch_target)

            test_loss.update(loss.item(), batch_target.size(0))
            test_acc.update(acc1[0], batch_target.size(0))

            y_true = np.concatenate((y_true, batch_target.cpu().detach().numpy(
            )), axis=0) if len(y_true) else batch_target.cpu().detach().numpy()
            y_pred = np.concatenate((y_pred, batch_pred.cpu().detach().numpy()), axis=0) if len(
                y_pred) else batch_pred.cpu().detach().numpy()
            penultimates = np.concatenate((penultimates, penultimate_pred.cpu().detach().numpy()), axis=0) if len(
                penultimates) else penultimate_pred.cpu().detach().numpy()  # Forvisulaization only

    test_auroc = roc_auc_score(
        y_true=y_true, y_score=softmax(y_pred, axis=1)[:, 1])
    test_precision = precision_score(
        y_true=y_true, y_pred=np.argmax(y_pred, axis=1))
    test_recall = recall_score(y_true=y_true, y_pred=np.argmax(y_pred, axis=1))
    test_f1 = f1_score(y_true=y_true, y_pred=np.argmax(y_pred, axis=1))

    return test_loss.avg, test_acc.avg, test_auroc, test_precision, test_recall, test_f1, y_true, y_pred, penultimates


def train_and_val(epoch, train_loader, val_loader,
                  model, criterion,  optimizer,
                  device, device_ids, lr_scheduler, patience_cnt,
                  best_acc,  checkpoint_dir, config, **kwargs):
    """One epoch training with validation inside"""
    model.train()
    train_acc = AverageMeter()
    train_auroc = AverageMeter()
    train_loss = AverageMeter()
    val_acc = val_auroc = val_loss = 0.0
    warmup_cond = config.TRAIN.get(
        "warming_up") is None or epoch >= config.TRAIN.get("warming_up")
    early_stop = False
    disp = {}
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Processing the mini-batch of data
        data_modality = [k for k in batch_data.keys() if k != "labels"]
        for k in batch_data.keys():
            if k in data_modality:
                batch_data[k] = batch_data[k].to(device=device, dtype=torch.float32)
            else:
                batch_data[k] = batch_data[k].to(device=device, dtype=torch.long)  
        random.shuffle(data_modality)
        batch_input = torch.cat(
            [batch_data[k] for k in data_modality], dim=0).contiguous()  # what ever key is
        batch_target = batch_data["labels"]
        batch_target = torch.cat(
            [batch_target]*len(data_modality), dim=0).contiguous()
        optimizer.zero_grad()
        if config.model in ["resnet50", "resnet18","resnet34"]:        
            batch_pred = model(batch_input)
            loss = criterion(batch_pred, batch_target)
        elif  'hkrawp' in config.model:
            batch_size = batch_input.size(0)
            n_modality=len(config.data_quality)
            n_base = batch_size // n_modality
            if epoch >= config.TRAIN.AWP.awp_warmup:
     
                awp = kwargs.get("awp").calc_awp(inputs_compressed=batch_input[:], # the best
                                            inputs_clean=batch_input[:], # the worst
                                            targets=batch_target[:],
                                            beta=6)
                kwargs.get("awp").perturb(awp)

            optimizer.zero_grad()
            ft1, ft2, ft3, ft4, feat, batch_pred = model(batch_input, True)
            logit_loss = criterion[0](batch_pred, batch_target)
            reg = criterion[1]([ft1, ft2, ft3, ft4, feat, batch_pred])
            loss =  config.TRAIN.alpha_logit * logit_loss  + config.TRAIN.HKR.alpha_hkr * reg
            disp["XELoss"] = logit_loss.item()
            disp["HKReg"] = reg.item()

        elif  'hkr' in config.model:
            ft1, ft2, ft3, ft4, feat, batch_pred = model(batch_input, True)
            logit_loss = criterion[0](batch_pred, batch_target)
            reg = criterion[1]([ft1, ft2, ft3, ft4, feat, batch_pred])

            loss =  config.TRAIN.alpha_logit * logit_loss  + config.TRAIN.HKR.alpha_hkr * reg
            disp["XELoss"] = logit_loss.item()
            disp["HKReg"] = reg.item()

        elif  "awp" in config.model:
            batch_size = batch_input.size(0)
            n_modality=len(config.data_quality)
            n_base = batch_size // n_modality
            if epoch >= config.TRAIN.AWP.awp_warmup:
                # list_mod = np.random.choice(n_modality, 2, replace=False)
                # list_mod.sort()
                awp = kwargs.get("awp").calc_awp(inputs_compressed=batch_input[:], # the best
                                            inputs_clean=batch_input[:], # the worst
                                            targets=batch_target[:],
                                            beta=6)
                kwargs.get("awp").perturb(awp)

            optimizer.zero_grad()
            batch_pred = model(batch_input)
            loss = F.cross_entropy(batch_pred, batch_target)
        else:
            raise ValueError("Criterion for model have not implemeted yet ...")

        loss.backward()
        optimizer.step()

        if ("hkrawp" in  config.model) or ("awp" in  config.model):
            if epoch >= config.TRAIN.AWP.awp_warmup:
                kwargs.get("awp").restore(awp)


        if config.TRAIN.scheduler == "one_cycle":
            lr_scheduler.step()

        acc1 = accuracy(batch_pred, batch_target)
        try: 
            auc = roc_auc_score(batch_target.cpu().detach().numpy(),
                            batch_pred.cpu().detach().numpy()[:, 1])
            
        except: 
            print("All single value ... ")
            auc = 0.5 # Only one class present
        train_loss.update(loss.item(), batch_target.size(0))
        train_acc.update(acc1[0], batch_target.size(0))
        train_auroc.update(auc, batch_target.size(0))

        sys.stdout.write("\r") 
        sys.stdout.flush()
        call = ' | '.join(["{}: {:.4f}".format(k, v) for k, v in disp.items()])
        sys.stdout.write("Train Epoch: {e:03d} Batch: {batch:05d}/{size:05d} | Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f} |{call}"
                         .format(e=epoch, batch=batch_idx+1, size=len(train_loader), loss=train_loss.avg, acc=train_acc.avg, auc=train_auroc.avg, call=call))

        if (batch_idx % (len(train_loader) // 10) == 0) and warmup_cond:

            sys.stdout.write("\n")
            val_loss, val_acc,  val_auroc,  val_precision, val_recall, val_f1 = validate(
                val_loader, model, torch.nn.CrossEntropyLoss(), device)
            print("Val loss: {loss:.4f} | Val acc: {acc:.4f} | Val AUC: {auc:.4f} | Val Pre: {pre:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}"
                  .format(loss=val_loss, acc=val_acc, auc=val_auroc, pre=val_precision, recall=val_recall, f1=val_f1))
            best = False
            if val_acc > best_acc:
                print(
                    f"Val Acc {bcolors.OKGREEN} improved {bcolors.ENDC} from {best_acc:.4f} to {val_acc:.4f} ")
                best_acc = val_acc
                best = True
                patience_cnt = 0
            else:
                print(
                    f"Val Acc does{bcolors.WARNING} NOT {bcolors.ENDC} improve from {best_acc:.4f}")
                patience_cnt += 1

            save_model(os.path.join(checkpoint_dir, config.running_name),
                       epoch, model, optimizer, lr_scheduler, device_ids, best)
            model.train()
        if config.TRAIN.apply_earlystop and patience_cnt >= config.TRAIN.patience:
            early_stop = True
            break

    return train_acc.avg, train_auroc.avg, train_loss.avg, val_acc, val_auroc, val_loss, best_acc, patience_cnt, early_stop
