from model_zoo import *
from helper.utils import (
    _seed_everything,
    setup_device,
    Logger,
    bcolors,
    set_trainable
)
from collections import OrderedDict
from helper.dataloader import get_dataloader
from helper.loops import train_and_val, test
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torchvision.transforms as transforms
import torch
from datetime import timedelta
from time import time
import argparse
import yaml
import json
import easydict
import os

import warnings
warnings.filterwarnings("ignore")
_seed_everything()

# args for config file
parser = argparse.ArgumentParser(description='Training depfake detection')
parser.add_argument('--config', type=str,
                    help='configuration file')


def _create_data_infor(dataset: str, qualities: list, test_qualities: list = []):
    """Return a dictionnary that include keys: train - val - test"""
    print("LOADING: ", dataset)
    with open(f"./dataset/{dataset}/data.json") as json_file:
        data = json.load(json_file)
    set_types = data.keys()
    if not test_qualities:
        test_qualities = qualities

    out_data = OrderedDict()
    for set_type in set_types: # train - val -test
        print(f"{set_type.upper()}ing set:")
        out_data[set_type] = OrderedDict()
        all_keys = data[set_type].keys()
        for q in all_keys: # walk through all qualities
            if set_type in ["train", "val"]:
                if q in qualities or q == "labels":
                    print(f"\tSelect {q}")
                    out_data[set_type][q] = data[set_type][q]
                    print(out_data[set_type][q][0])
            else:
                if q in test_qualities or q == "labels":
                    print(f"\tSelect {q}")
                    out_data[set_type][q] = data[set_type][q]
                    print(out_data[set_type][q][0])
    return out_data


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.SYS.GPUs
    device, device_ids = setup_device(config.SYS.num_gpus)
    checkpoint_dir = os.path.join(config.output_folder,
                                  config.dataset,
                                  config.model,
                                  config.running_name,
                                  'checkpoints')

    logging_dir = os.path.join(config.output_folder,
                               config.dataset,
                               config.model,
                               config.running_name,
                               'logs')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # define the model
    print(f"Network training: {bcolors.OKBLUE} {config.model} {bcolors.ENDC}")
    modules = OrderedDict()
    

    if "hkrawp" in config.model:
        print(f"{bcolors.HEADER}AWP w/ HSIC {bcolors.ENDC}")
        model = resnet50(pretrained=True,  num_classes=2).to(device)
        set_trainable(model, False, ['fc'], [0])  # for  warming up
     
        criterion = torch.nn.ModuleList([])
        criterion.append(torch.nn.CrossEntropyLoss().cuda())
        criterion.append(HKRPair(weights=config.TRAIN.HKR.weights, 
                                n_modality=len(config.data_quality), 
                                sigma=config.TRAIN.HKR.sigma))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        param_dict = [{'params': model.parameters(), 'lr': config.TRAIN.lr}]

        proxy = resnet50(pretrained=True,  num_classes=2).to(device)
        proxy_optim = optim.SGD(proxy.parameters(), lr=config.TRAIN.lr)
        modules["awp"] =  AWP(model=torch.nn.DataParallel(model, device_ids=device_ids), 
                                    proxy=proxy, proxy_optim=proxy_optim, 
                                    gamma=config.TRAIN.AWP.awp_gamma)


    elif 'hkr' in config.model:
        print(f"{bcolors.HEADER}Apply Hilbert Kernel Regularization{bcolors.ENDC}")
        model = resnet50(pretrained=True,  num_classes=2)
        set_trainable(model=model, boolean=False, except_layers=['fc'], device_ids=[0])
   
        
        criterion = torch.nn.ModuleList([])
        criterion.append(torch.nn.CrossEntropyLoss().cuda())

        criterion.append(HKRPair(weights=config.TRAIN.HKR.weights, 
                                n_modality=len(config.data_quality), 
                                sigma=config.TRAIN.HKR.sigma))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        param_dict = [{'params': model.parameters(), 'lr': config.TRAIN.lr}]

    elif "awp" in config.model:
        print(f"{bcolors.HEADER}Adversarial weight purturbation{bcolors.ENDC}")
        model = resnet50(pretrained=True,  num_classes=2).to(device)
        set_trainable(model, False, ['fc'], [0])  # for  warming up

        criterion = torch.nn.CrossEntropyLoss()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        param_dict = [{'params': model.parameters(), 'lr': config.TRAIN.lr}]

        proxy = resnet50(pretrained=True,  num_classes=2).to(device)
        proxy_optim = optim.SGD(proxy.parameters(), lr=config.TRAIN.lr)
        modules["awp"] =  AWP(model=torch.nn.DataParallel(model, device_ids=device_ids), 
                                    proxy=proxy, proxy_optim=proxy_optim, 
                                    gamma=config.TRAIN.AWP.awp_gamma)
    
    else:
        raise ValueError(f"{config.model} is not exist!")

    model = model.to(device)
    if (len(device_ids) > 1) and (not isinstance(model, torch.nn.DataParallel)):
        model = torch.nn.DataParallel(model, device_ids=device_ids)
   

    data_info = _create_data_infor(dataset=config.dataset if config.get("test_dataset") is None else config.test_dataset, # For test phase
                                    qualities=config.data_quality,
                                    test_qualities=config.test_data_quality)
    
    train_loader, val_loader, test_loader = get_dataloader(data_info=data_info,
                                                            dataset_name=config.dataset,
                                                            normalization=normalize,
                                                            image_size=config.TRAIN.image_size,
                                                            batch_size=config.TRAIN.batch_size,
                                                            num_workers=config.SYS.num_workers,
                                                            return_pair=not config.data_shuffle,
                                                            aug=None, 
                                                            intra=config.TRAIN.get("intra"),
                                                            test_random_compression=config.get("test_random_compress"))

    if config.is_train:
        with open(logging_dir + '/config.yaml', 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

        logger = Logger(os.path.join(
            logging_dir, config.running_name + '_train_val.txt'))
        logger.set_names(['Epoch', 'lr',
                          'Train Loss', 'Train Acc.', 'Train Auc.',
                          'Val Loss',  'Val Acc.', 'Val Auc.'])

        # define optimizer and learing rate schedule
        print(
            f"Optimizer: {bcolors.OKBLUE} {config.TRAIN.optimizer} {bcolors.ENDC} ")
        if config.TRAIN.optimizer == 'Adam':
            optimizer = optim.Adam(
                param_dict,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=config.TRAIN.weight_decay,
                amsgrad=False)

        elif config.TRAIN.optimizer == 'SGD':
            optimizer = optim.SGD(param_dict,
                                  momentum=config.TRAIN.momentum,
                                  weight_decay=config.TRAIN.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer { config.TRAIN.optimizer}")

        print(
            f"Learning rate scheduler: {bcolors.OKBLUE} {config.TRAIN.scheduler} {bcolors.ENDC} ")
        if config.TRAIN.scheduler == "cosine":
            lr_scheduler = CosineAnnealingLR(optimizer=optimizer,
                                             T_max=10,
                                             eta_min=1e-5,
                                             verbose=True)
        elif config.TRAIN.scheduler == "one_cycle":
            lr_scheduler = OneCycleLR(optimizer=optimizer,
                                      max_lr=config.TRAIN.lr,
                                      epochs=config.TRAIN.epochs,
                                      steps_per_epoch=len(train_loader),
                                      pct_start=0.05,
                                      total_steps=None,
                                      verbose=False)
        else:
            raise ValueError(f"Unknown scheduler: {config.TRAIN.scheduler}")

        # Training PHASE
        best_acc = 0.0
        early_stop = False
        patience_cnt = 0
        tic_e = time()
        for epoch in range(config.TRAIN.epochs):
            if config.TRAIN.get("warming_up") == epoch:
                set_trainable(model.module if len(device_ids) > 1 else model, True, [], device_ids)
  
            train_acc, train_auroc, train_loss, \
                val_acc, val_auroc, val_loss,\
                best_acc, patience_cnt, early_stop = train_and_val(epoch, train_loader, val_loader,
                                                                    model, criterion, optimizer, device,
                                                                    device_ids, lr_scheduler, patience_cnt,
                                                                    best_acc, checkpoint_dir, config, **modules)
            if not config.TRAIN.scheduler == "one_cycle":
                lr_scheduler.step()
            logger.append([epoch, optimizer.param_groups[0]['lr'],
                           train_loss, train_acc, train_auroc,
                           val_loss, val_acc, val_auroc])

            eta = str(timedelta(seconds=int(
                (time()-tic_e) / (epoch+1) * (config.TRAIN.epochs))))
            print('\n============== ETA: {} =============='.format(eta))

            if early_stop:
                print(f"{bcolors.WARNING}Early stopping {bcolors.ENDC}")
                break
            if  "matnet" in config.model:
                if len(device_ids) > 1:
                    model.module.auxiliary_loss.alpha *= config.TRAIN.alpha_decay
                else:
                    model.auxiliary_loss.alpha *= config.TRAIN.alpha_decay
        logger.close()

    # Test PHASE
    pretrained_weights = torch.load(str('_'.join(
        [os.path.join(checkpoint_dir, config.running_name), 'best.pth'])))['state_dict']
    print("Load model from: ", str(
        '_'.join([os.path.join(checkpoint_dir, config.running_name), 'best.pth'])))
    if len(device_ids) > 1:
        model = model.module
    model.load_state_dict(pretrained_weights)

    test_loss, test_acc, test_auroc, \
        test_precision, test_recall, test_f1, \
        test_true, test_pred, penultimates = test(
            test_loader, model, torch.nn.CrossEntropyLoss(), device)
    print(f"{bcolors.OKGREEN} Test loss: {test_loss:.4f} "
          f"\n Test ACC:\t {test_acc:.4f} "
          f"\n Test AUC:\t {test_auroc:.4f} "
          f"\n Test Pre:\t {test_precision:.4f} "
          f"\n Test Rec:\t {test_recall:.4f} "
          f"\n Test F1:\t {test_f1:.4f} {bcolors.ENDC}"
          )

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as cf:
        config = yaml.safe_load(cf)
    config = easydict.EasyDict(config)
    print(vars(config))
    main(config)

 