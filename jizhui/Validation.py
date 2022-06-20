import os
import sys
import argparse
import importlib
import shutil
import imageio
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import albumentations as albu

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_path not in sys.path:
    sys.path.append(project_path)


from seg_pipeline.Dataset import SegDataset, get_k_fold_cross_validation_df
from seg_pipeline.utils.helpers import load_yaml, init_seed, init_logger
from seg_pipeline.Learning import Learning


def argparser():
    parser = argparse.ArgumentParser(description='segmentation pipeline')
    parser.add_argument('-cfg',
                        default=os.path.join(project_path, 'seg_pipeline/experiments/seunet_multi/train_config_512.yaml'),
                        type=str, help='experiment name')
    parser.add_argument('-fold_id', default=0, type=int)
    parser.add_argument('-use_flip', default=False, type=bool)
    return parser.parse_args()


def validate_image(model, images, device):
    with torch.no_grad():
        images = images.to(device)
        predicted = model(images)
        # predicted = torch.softmax(predicted, dim=1)
        predicted = torch.sigmoid(predicted)
    return predicted.cpu().detach().numpy()


def make_gif_plus(image, label, pred, gif_path):
    def convert_channel(original_data, label_data, value, channel, convert_value):
        for i in range(0, 3):
            if i == channel:
                original_data[label_data == value, channel] = convert_value

    image = image.convert('RGB')
    original_data = np.array(image)

    convert_data_mask = original_data.copy()
    label_data = np.array(label)

    convert_channel(convert_data_mask, label_data[:, :, 0], 1, 0, 255)
    convert_channel(convert_data_mask, label_data[:, :, 0], 1, 1, 255)
    convert_channel(convert_data_mask, label_data[:, :, 0], 1, 2, 0)
    convert_channel(convert_data_mask, label_data[:, :, 1], 1, 0, 255)
    convert_channel(convert_data_mask, label_data[:, :, 1], 1, 1, 0)
    convert_channel(convert_data_mask, label_data[:, :, 1], 1, 2, 0)
    convert_channel(convert_data_mask, label_data[:, :, 2], 1, 0, 0)
    convert_channel(convert_data_mask, label_data[:, :, 2], 1, 1, 255)
    convert_channel(convert_data_mask, label_data[:, :, 2], 1, 2, 0)
    convert_channel(convert_data_mask, label_data[:, :, 3], 1, 0, 0)
    convert_channel(convert_data_mask, label_data[:, :, 3], 1, 1, 0)
    convert_channel(convert_data_mask, label_data[:, :, 3], 1, 2, 255)
    convert_channel(convert_data_mask, label_data[:, :, 4], 1, 0, 255)
    convert_channel(convert_data_mask, label_data[:, :, 4], 1, 1, 0)
    convert_channel(convert_data_mask, label_data[:, :, 4], 1, 2, 255)
    convert_channel(convert_data_mask, label_data[:, :, 5], 1, 0, 0)
    convert_channel(convert_data_mask, label_data[:, :, 5], 1, 1, 255)
    convert_channel(convert_data_mask, label_data[:, :, 5], 1, 2, 255)
    # convert_channel(convert_data_mask, label_data[:, :, 6], 1, 0, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 6], 1, 1, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 6], 1, 2, 255)
    # convert_channel(convert_data_mask, label_data[:, :, 7], 1, 0, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 7], 1, 1, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 7], 1, 2, 255)
    # convert_channel(convert_data_mask, label_data[:, :, 8], 1, 0, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 8], 1, 1, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 8], 1, 2, 255)
    # convert_channel(convert_data_mask, label_data[:, :, 9], 1, 0, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 9], 1, 1, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 9], 1, 2, 255)
    # convert_channel(convert_data_mask, label_data[:, :, 10], 1, 0, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 10], 1, 1, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 10], 1, 2, 255)
    # convert_channel(convert_data_mask, label_data[:, :, 11], 1, 0, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 11], 1, 1, 0)
    # convert_channel(convert_data_mask, label_data[:, :, 11], 1, 2, 255)


    convert_data_pred = original_data.copy()
    pred_data = np.array(pred)

    convert_channel(convert_data_pred, pred_data[0, :, :], 1, 0, 255)
    convert_channel(convert_data_pred, pred_data[0, :, :], 1, 1, 255)
    convert_channel(convert_data_pred, pred_data[0, :, :], 1, 2, 0)
    convert_channel(convert_data_pred, pred_data[1, :, :], 1, 0, 255)
    convert_channel(convert_data_pred, pred_data[1, :, :], 1, 1, 0)
    convert_channel(convert_data_pred, pred_data[1, :, :], 1, 2, 0)
    convert_channel(convert_data_pred, pred_data[2, :, :], 1, 0, 0)
    convert_channel(convert_data_pred, pred_data[2, :, :], 1, 1, 255)
    convert_channel(convert_data_pred, pred_data[2, :, :], 1, 2, 0)
    convert_channel(convert_data_pred, pred_data[3, :, :], 1, 0, 0)
    convert_channel(convert_data_pred, pred_data[3, :, :], 1, 1, 0)
    convert_channel(convert_data_pred, pred_data[3, :, :], 1, 2, 255)
    convert_channel(convert_data_pred, pred_data[4, :, :], 1, 0, 255)
    convert_channel(convert_data_pred, pred_data[4, :, :], 1, 1, 0)
    convert_channel(convert_data_pred, pred_data[4, :, :], 1, 2, 255)
    convert_channel(convert_data_pred, pred_data[5, :, :], 1, 0, 0)
    convert_channel(convert_data_pred, pred_data[5, :, :], 1, 1, 255)
    convert_channel(convert_data_pred, pred_data[5, :, :], 1, 2, 255)
    # convert_channel(convert_data_pred, pred_data[6, :, :], 1, 0, 0)
    # convert_channel(convert_data_pred, pred_data[6, :, :], 1, 1, 0)
    # convert_channel(convert_data_pred, pred_data[6, :, :], 1, 2, 255)
    # convert_channel(convert_data_pred, pred_data[7, :, :], 1, 0, 0)
    # convert_channel(convert_data_pred, pred_data[7, :, :], 1, 1, 0)
    # convert_channel(convert_data_pred, pred_data[7, :, :], 1, 2, 255)
    # convert_channel(convert_data_pred, pred_data[8, :, :], 1, 0, 0)
    # convert_channel(convert_data_pred, pred_data[8, :, :], 1, 1, 0)
    # convert_channel(convert_data_pred, pred_data[8, :, :], 1, 2, 255)
    # convert_channel(convert_data_pred, pred_data[9, :, :], 1, 0, 0)
    # convert_channel(convert_data_pred, pred_data[9, :, :], 1, 1, 0)
    # convert_channel(convert_data_pred, pred_data[9, :, :], 1, 2, 255)
    # convert_channel(convert_data_pred, pred_data[10, :, :], 1, 0, 0)
    # convert_channel(convert_data_pred, pred_data[10, :, :], 1, 1, 0)
    # convert_channel(convert_data_pred, pred_data[10, :, :], 1, 2, 255)
    # convert_channel(convert_data_pred, pred_data[11, :, :], 1, 0, 0)
    # convert_channel(convert_data_pred, pred_data[11, :, :], 1, 1, 0)
    # convert_channel(convert_data_pred, pred_data[11, :, :], 1, 2, 255)

    original_data=np.array(Image.fromarray(original_data).transpose(Image.FLIP_TOP_BOTTOM))
    convert_data_mask=np.array(Image.fromarray(convert_data_mask).transpose(Image.FLIP_TOP_BOTTOM))
    convert_data_pred=np.array(Image.fromarray(convert_data_pred).transpose(Image.FLIP_TOP_BOTTOM))
    frames = [original_data, convert_data_mask, convert_data_pred]
    imageio.mimsave(gif_path, frames, 'GIF', duration=1)


def validate_model(model, loader, device, use_flip, seg_classes_num, result_path):
    current_score_mean = 0
    # current_score_each_class = np.zeros((seg_classes_num - 1, ))
    current_score_each_class = np.zeros((seg_classes_num,))
    info_all = {}
    for batch_idx, (image_ids, images, labels) in enumerate(tqdm(loader)):
        predicts = validate_image(model, images, device)
        labels = labels.numpy()
        if use_flip:
            flipped_imgs = torch.flip(images, dims=(3,))
            flipped_predicts = validate_image(model, flipped_imgs, device)
            flipped_predicts = np.flip(flipped_predicts, axis=2)
            predicts = (predicts + flipped_predicts) / 2

        # calculate validation score
        score_list = []
        masks = np.argmax(predicts, axis=1)
        masks = (predicts > 0.5).astype(np.uint8)
        # for i_label in range(1, seg_classes_num):
        for i_label in range(0, seg_classes_num):
            # score_list.append(dice_score_fn(masks, labels, i_label))
            score_list.append(Learning.dice_score_fn(masks[:, i_label], labels[..., i_label], 1))

        concat_score = np.array(score_list)

        score, score_each_class = np.mean(concat_score), np.mean(concat_score, axis=1)
        current_score_mean = (current_score_mean * batch_idx + score) / (batch_idx + 1)
        current_score_each_class = (current_score_each_class * batch_idx + score_each_class) / (batch_idx + 1)

        # make gif
        for name, image, label, mask in zip(image_ids, images, labels, masks):
            img_arr = image.detach().numpy()
            label_arr = label

            img_arr = np.asarray(img_arr[0, :, :] * 255 + 0.5, dtype=np.uint8)
            label_arr = np.asarray(label_arr, dtype=np.uint8)
            mask_arr = np.asarray(mask, dtype=np.uint8)

            im_pil = Image.fromarray(img_arr).convert('RGB')

            make_gif_plus(im_pil, label_arr, mask_arr, os.path.join(result_path, name+'.gif'))
            info_all.update({name: {
                'img_arr': img_arr,
                'label_arr': label_arr,
                'mask_arr': mask_arr
                }
            })

    msg = 'score: {:.6f}, score_each_class: ' + '{:.4f},' * (seg_classes_num)
    msg = msg.format(current_score_mean, *current_score_each_class)
    with open('info_val.pkl', 'wb') as p_w:
        pickle.dump(info_all, p_w)
    print(msg)


def main():
    args = argparser()
    config_path = args.cfg
    fold_id = args.fold_id
    use_flip = args.use_flip
    experiment_folder = os.path.dirname(config_path)
    validation_config = load_yaml(config_path)
    print(validation_config)

    seed = validation_config['MAIN']['SEED']
    scale_mode = validation_config['MAIN']['SCALE']
    init_seed(seed)

    batch_size = validation_config['VALIDATION']['BATCH_SIZE']
    device = validation_config['MAIN']['DEVICE']
    if "DEVICE_LIST" in validation_config['MAIN']:
        # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, validation_config['MAIN']["DEVICE_LIST"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    module = importlib.import_module(validation_config['MODEL']['PY'])
    model_class = getattr(module, validation_config['MODEL']['CLASS'])
    model = model_class(**validation_config['MODEL'].get('ARGS', None)).to(device)

    num_workers = validation_config['MAIN']['WORKERS']
    valid_transform = albu.load(validation_config['DATA']['VALID_TRANSFORMS'])
    dataset_folder = validation_config['DATA']['DATA_DIRECTORY']
    dataset_extra = validation_config['DATA']['DATA_EXTRA']
    img_size = tuple(validation_config['DATA']['GENERATE_IMG_SIZE'])
    n_folds = validation_config['DATA']['FOLD_NUMBER']
    seg_classes_num = validation_config['MODEL']['ARGS']['seg_classes']

    folds_data_info_df = get_k_fold_cross_validation_df(dataset_folder, dataset_extra, n_folds)

    valid_dataset = SegDataset(
        data_folder=dataset_folder, img_size=img_size, mode='val',
        data_transform=valid_transform, fold_index=fold_id,
        folds_data_info_df=folds_data_info_df, scale=scale_mode
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size,
        num_workers=0, shuffle=False
    )

    model_config = validation_config['MODEL'].get('PRETRAINED', False)
    loaded_pipeline_name = model_config['PIPELINE_NAME']
    checkpoint_path = Path(
        model_config['PIPELINE_PATH'],
        model_config['CHECKPOINTS_FOLDER'],
            '{}_fold_{}.pth'.format(loaded_pipeline_name, fold_id)
        )

    if 'VALIDATION_RESULT_FOLDER' in validation_config['VALIDATION']:
        result_path = validation_config['VALIDATION']['VALIDATION_RESULT_FOLDER']
    else:
        result_path = Path(experiment_folder, 'val_results_', 'fold_'+str(fold_id))

    if os.path.isdir(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    model.load_state_dict(torch.load(checkpoint_path))
    if len(validation_config['MAIN']['DEVICE_LIST']) > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    # summary(model, (3, 512, 512), batch_size=1, device='cuda')
    validate_model(model, valid_dataloader, device, use_flip, seg_classes_num, result_path)


if __name__ == "__main__":
    main()
