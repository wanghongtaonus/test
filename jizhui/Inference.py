import os
import sys
import argparse
import importlib
import shutil
import imageio
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import nibabel as nib


import torch
from torch.utils.data import DataLoader
import albumentations as albu

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_path not in sys.path:
    sys.path.append(project_path)


from seg_pipeline.Dataset import SegDataset
from seg_pipeline.utils.helpers import load_yaml, init_seed


def argparser():
    parser = argparse.ArgumentParser(description='segmentation pipeline')
    parser.add_argument('-cfg',
                        default=os.path.join(project_path, 'seg_pipeline/experiments/seunet_multi/train_config_512.yaml'),
                        type=str, help='experiment name')
    parser.add_argument('-fold_id', default=0, type=int)
    parser.add_argument('-use_flip', default=False, type=bool)
    parser.add_argument('-data_folder', default='/home/htwang/data/test', type=str)
    return parser.parse_args()


def validate_image(model, images, device):
    with torch.no_grad():
        images = images.to(device)
        predicted = model(images)
        predicted = torch.sigmoid(predicted)
    return predicted.cpu().detach().numpy()


def make_gif_plus(image, pred, gif_path):
    def convert_channel(original_data, label_data, value, channel, convert_value):
        for i in range(0, 3):
            if i == channel:
                original_data[label_data == value, channel] = convert_value

    image = image.convert('RGB')
    original_data = np.array(image)

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

    frames = [original_data, convert_data_pred]
    imageio.mimsave(gif_path, frames, 'GIF', duration=0.5)
def get_mid_index(mask):
    mask_mid_center = (np.nonzero(np.sum(mask, axis=0))[0][0] +
                       np.nonzero(np.sum(mask, axis=0))[0][-1]) // 2
    return mask_mid_center


def _save_mask_nii(pred, str_number, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if int(str_number) <= 12:
        _number = int(str_number)
    # elif int(str_number) == 7:
    #     _number = 4
    # elif int(str_number) == 8:
    #     _number = 5
    # elif int(str_number) == 9:
    #     _number = 6
    # else:
    #     _number = 7

    pred = pred.transpose(1, 0)
    pred = pred[:, :, np.newaxis]
    pred[pred == 1] = _number
    nib_file = nib.Nifti1Image(pred, np.eye(4))
    nib.loadsave.save(nib_file, save_path)


def save_mask_nii(pred, height, width, save_path, file_name):
    for _idx in range(6):
        _pred = pred[_idx, :, :]
        _pred = cv2.resize(_pred, (width.item(), height.item()), interpolation=cv2.INTER_NEAREST)
        _mid = get_mid_index(_pred)
        _left_pred = np.zeros_like(_pred)
        _left_pred[_pred == 1] = 1
        _left_pred[:, _mid:] = 0
        _right_pred = np.zeros_like(_pred)
        _right_pred[_pred == 1] = 1
        _right_pred[:, :_mid] = 0
        file_name=file_name.split('.')[0]
        _save_mask_nii(_left_pred, str(_idx + 1), os.path.join(save_path, file_name, file_name + '_' + str(_idx + 1) + '.nii.gz'))
        _save_mask_nii(_right_pred, str(_idx + 7), os.path.join(save_path, file_name, file_name + '_' + str(_idx + 7) + '.nii.gz'))


def infer_model(model, loader, device, use_flip, result_path):


    for batch_idx, (image_ids, images, heights, widths) in enumerate(tqdm(loader)):
        predicts = validate_image(model, images, device)
        if use_flip:
            flipped_imgs = torch.flip(images, dims=(3,))
            flipped_predicts = validate_image(model, flipped_imgs, device)
            flipped_predicts = np.flip(flipped_predicts, axis=2)
            predicts = (predicts + flipped_predicts) / 2
        masks = (predicts > 0.5).astype(np.uint8)
        # masks = np.array(Image.fromarray(masks).transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8)
        # masks = np.transpose(masks[0], (1, 2, 0))
        # tem=np.flipud(masks)
        # masks=np.transpose(tem,(2,0,1))
        # masks=masks[np.newaxis,:,:,:]


        # make gif
        for name, image, mask, height, width in zip(image_ids, images, masks, heights, widths):
            mask = np.transpose(mask, (1, 2, 0))
            tem = np.flipud(mask)
            mask = np.transpose(tem, (2, 0, 1))
            img_arr = image.detach().numpy()

            img_arr = np.asarray(img_arr[0, :, :] * 255 + 0.5, dtype=np.uint8)
            mask_arr = np.asarray(mask, dtype=np.uint8)

            # mask_arr = np.flipud(mask_arr)
            # im_pil = Image.fromarray(img_arr).convert('RGB')
            # make_gif_plus(im_pil, mask_arr, os.path.join(result_path, name+'.gif'))
            try:
                save_mask_nii(mask_arr, height, width, result_path, name)
            except:
                pass


def main():
    args = argparser()
    config_path = args.cfg
    fold_id = args.fold_id
    use_flip = args.use_flip
    data_folder = args.data_folder
    infer_config = load_yaml(config_path)
    print(infer_config)

    seed = infer_config['MAIN']['SEED']
    scale_mode = infer_config['MAIN']['SCALE']
    init_seed(seed)

    batch_size = infer_config['VALIDATION']['BATCH_SIZE']
    device = infer_config['MAIN']['DEVICE']
    if "DEVICE_LIST" in infer_config['MAIN']:
        # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, infer_config['MAIN']["DEVICE_LIST"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    module = importlib.import_module(infer_config['MODEL']['PY'])
    model_class = getattr(module, infer_config['MODEL']['CLASS'])
    model = model_class(**infer_config['MODEL'].get('ARGS', None)).to(device)

    valid_transform = albu.load(os.path.join(os.path.dirname(__file__), infer_config['DATA']['VALID_TRANSFORMS']))
    img_size = tuple(infer_config['DATA']['GENERATE_IMG_SIZE'])

    test_dataset = SegDataset(
        data_folder=data_folder, img_size=img_size, mode='test',
        data_transform=valid_transform, fold_index=fold_id, scale=scale_mode, data_format='mhd'
    )
    valid_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        num_workers=0, shuffle=False
    )

    model_config = infer_config['MODEL'].get('PRETRAINED', False)
    loaded_pipeline_name = model_config['PIPELINE_NAME']
    checkpoint_path = Path(
        model_config['PIPELINE_PATH'],
        model_config['CHECKPOINTS_FOLDER'], '{}_fold_{}.pth'.format(loaded_pipeline_name, fold_id))

    result_path = data_folder + '_result'
    # result_path = data_folder
    if os.path.isdir(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), checkpoint_path)))
    if len(infer_config['MAIN']['DEVICE_LIST']) > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    infer_model(model, valid_dataloader, device, use_flip, result_path)


if __name__ == "__main__":
    main()
