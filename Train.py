import os
import sys
import argparse
import importlib
from pathlib import Path

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
import albumentations as albu

from seg_pipeline.Dataset import SegDataset, get_k_fold_cross_validation_df
from seg_pipeline.Learning import Learning
from seg_pipeline.utils.helpers import load_yaml, init_seed, init_logger, get_config_info_tree
# from seg_pipeline import Losses as Losses


def argparser():
    parser = argparse.ArgumentParser(description='Segmentation pipeline')
    parser.add_argument('-train_cfg',
                        default=os.path.join(project_path, 'seg_pipeline/experiments/seunet_multi/train_config_512.yaml'),
                        type=str, help='train configs path')
    return parser.parse_args()


def train_fold(
    train_config, experiment_folder, pipeline_name, main_logger, log_dir, fold_id,
    train_dataloader, valid_dataloader,):

    fold_logger_dir = Path(log_dir, 'fold_'+str(fold_id))
    fold_logger_dir.mkdir(exist_ok=True, parents=True)
    fold_logger = init_logger(fold_logger_dir, 'train_fold_{}.log'.format(fold_id), use_stdout=False)  # sub logger file

    best_checkpoint_folder = Path(experiment_folder, train_config['TRAINING']['CHECKPOINTS']['BEST_FOLDER'])
    best_checkpoint_folder.mkdir(exist_ok=True, parents=True)

    checkpoints_history_folder = Path(
        experiment_folder,
        train_config['TRAINING']['CHECKPOINTS']['FULL_FOLDER'],
        'fold_{}'.format(fold_id)
    )
    checkpoints_history_folder.mkdir(exist_ok=True, parents=True)
    checkpoints_topk = train_config['TRAINING']['CHECKPOINTS']['TOPK']

    calculation_name = '{}_fold_{}'.format(pipeline_name, fold_id)
    
    device = train_config['MAIN']['DEVICE']
    use_fp_16 = train_config['MAIN']['USE_FP16']

    module = importlib.import_module(train_config['MODEL']['PY'])
    model_class = getattr(module, train_config['MODEL']['CLASS'])
    model = model_class(**train_config['MODEL']['ARGS'])
    model.to(device)

    seg_classes_num = train_config['MODEL']['ARGS']['seg_classes']

    optimizer_class = getattr(torch.optim, train_config['TRAINING']['OPTIMIZER']['CLASS'])
    optimizer = optimizer_class(model.parameters(), **train_config['TRAINING']['OPTIMIZER']['ARGS'])
    scheduler_class = getattr(torch.optim.lr_scheduler, train_config['TRAINING']['SCHEDULER']['CLASS'])
    scheduler = scheduler_class(optimizer, **train_config['TRAINING']['SCHEDULER']['ARGS'])

    pretrained_model_config = train_config['MODEL'].get('PRETRAINED', False)
    if pretrained_model_config: 
        loaded_pipeline_name = pretrained_model_config['PIPELINE_NAME']
        pretrained_model_path = Path(
            pretrained_model_config['PIPELINE_PATH'], 
            pretrained_model_config['CHECKPOINTS_FOLDER'],
            '{}_fold_{}.pth'.format(loaded_pipeline_name, fold_id)
        ) 
        if pretrained_model_path.is_file():
            pre_state_dict = torch.load(pretrained_model_path)
            pre_state_dict = {k: pre_state_dict[k] for k in pre_state_dict.keys() if 'final' not in str(k)}
            model.load_state_dict(pre_state_dict, strict=False)
            fold_logger.info('load model from {}'.format(pretrained_model_path))
            main_logger.info('load model from {}'.format(pretrained_model_path))

    if len(train_config['MAIN']['DEVICE_LIST']) > 1:
        model = torch.nn.DataParallel(model)
    
    # module = importlib.import_module(train_config['CRITERION']['PY'])
    # loss_class = getattr(Losses, train_config['TRAINING']['CRITERION']['CLASS'])
    # loss_fn = loss_class(**train_config['TRAINING']['CRITERION']['ARGS'])
    # todo need more losses
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = train_config['TRAINING']['CRITERION']['CLASS']
    loss_args = train_config['TRAINING']['CRITERION']['ARGS']

    n_epoches = train_config['TRAINING']['EPOCHES']
    grad_clip = train_config['TRAINING']['GRADIENT_CLIPPING']
    grad_accum = train_config['TRAINING']['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = train_config['TRAINING']['EARLY_STOPPING']
    validation_frequency = train_config['TRAINING'].get('VALIDATION_FREQUENCY', 1)
    use_swa = train_config['TRAINING']['USE_SWA']

    freeze_model = train_config['MODEL']['FREEZE']

    # make tensorboard
    if train_config['TRAINING']['USE_TF_BOARD']:
        tf_logger = SummaryWriter(Path(fold_logger_dir, 'tf_logger'))
    else:
        tf_logger = None
    
    Learning(
        optimizer,
        loss_fn,
        loss_args,
        seg_classes_num,
        device,
        n_epoches,
        scheduler,
        freeze_model,
        grad_clip,
        grad_accum,
        early_stopping,
        validation_frequency,
        calculation_name,
        best_checkpoint_folder,
        checkpoints_history_folder,
        checkpoints_topk,
        main_logger,
        fold_logger,
        tf_logger=tf_logger,
        use_fp_16=use_fp_16,
        use_swa=use_swa,
    ).run_train(model, train_dataloader, valid_dataloader)

    if train_config['TRAINING']['USE_TF_BOARD']:
        tf_logger.close()


def main():
    args = argparser()
    config_folder = args.train_cfg
    experiment_folder = os.path.dirname(config_folder)
    train_config = load_yaml(config_folder)

    log_dir = Path(experiment_folder, train_config['MAIN']['LOGGER_DIR'])
    log_dir.mkdir(exist_ok=True, parents=True)

    main_logger = init_logger(log_dir, 'train_main.log')

    seed = train_config['MAIN']['SEED']
    scale_mode = train_config['MAIN']['SCALE']
    init_seed(seed)
    train_config_info = get_config_info_tree(train_config)
    main_logger.info(train_config_info)

    if "DEVICE_LIST" in train_config['MAIN']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, train_config['MAIN']['DEVICE_LIST']))

    pipeline_name = train_config['MAIN']['PIPELINE_NAME']
    num_workers = train_config['MAIN']['WORKERS']
    batch_size = train_config['TRAINING']['BATCH_SIZE']

    train_transform = albu.load(train_config['DATA']['TRAIN_TRANSFORMS'])
    valid_transform = albu.load(train_config['DATA']['VALID_TRANSFORMS'])

    use_sampler = train_config['DATA']['USE_SAMPLER']

    dataset_folder = train_config['DATA']['DATA_DIRECTORY']
    dataset_extra = train_config['DATA']['DATA_EXTRA']
    img_size = tuple(train_config['DATA']['GENERATE_IMG_SIZE'])
    n_folds = train_config['DATA']['FOLD_NUMBER']
    usefolds = train_config['DATA']['USE_FOLDS']

    if len(usefolds) > n_folds:
        raise ValueError('length of usefolds must be lower than n_folds')

    folds_data_info_df = get_k_fold_cross_validation_df(dataset_folder, dataset_extra, n_folds)

    for fold_id in usefolds:
        main_logger.info('Start training of {} fold...'.format(fold_id))

        train_dataset = SegDataset(
            data_folder=dataset_folder, img_size=img_size, mode='train',
            data_transform=train_transform, fold_index=fold_id,
            folds_data_info_df=folds_data_info_df, scale=scale_mode
        )
        train_sampler = None
        if use_sampler:
            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=batch_size,   
                num_workers=num_workers, sampler=train_sampler
            )
        else:
            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=batch_size,   
                num_workers=num_workers, shuffle=True
            )

        valid_dataset = SegDataset(
            data_folder=dataset_folder, img_size=img_size, mode='val',
            data_transform=valid_transform, fold_index=fold_id,
            folds_data_info_df=folds_data_info_df,
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset, batch_size=batch_size, 
            num_workers=num_workers, shuffle=False
        )

        train_fold(
            train_config, experiment_folder, pipeline_name, main_logger, log_dir, fold_id,
            train_dataloader, valid_dataloader,
        )


if __name__ == "__main__":
    main()
