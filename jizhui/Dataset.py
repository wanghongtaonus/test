import os
import math
import random

import cv2
import pydicom
import numpy as np
import pandas as pd
import nibabel as nib

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transform
import md.image3d.python.image3d_io as cio
from md.image3d.python.image3d import Image3d

special_data_list = [
    'CR202101100247_R7.nii.gz'
]


class SegDataset(Dataset):
    def __init__(self, img_size, mode, data_transform=None, fold_index=None, folds_data_info_df=None, data_folder=None,
                 scale='coarse', data_format='mhd'):
        
        self.transform = data_transform
        self.mode = mode
        self.scale = scale
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.data_format = data_format

        self.train_image_path = data_folder
        if mode == 'train' or mode == 'val':
            assert fold_index is not None
            assert folds_data_info_df is not None

        if mode == 'test':
            assert data_folder is not None
            self.test_image_path = data_folder

        self.case_list = None
        self.data_file_list = None
        self.data_folder_list = None
        self.data_type_list = None
        self.key_start_points_dict = {}
        self.num_data = 0
        self.fold_index = None
        self.folds_data_info_df = folds_data_info_df
        self.set_mode(mode, fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index

        if self.mode == 'train':
            folds = self.folds_data_info_df
            folds = folds[folds.fold != fold_index]
            
            self.case_list = folds.case_name.values.tolist()
            self.data_file_list = folds.train_file_path.values.tolist()
            self.data_folder_list = folds.train_file_folder.values.tolist()
            self.data_type_list = folds.type.tolist()
            self.num_data = len(self.case_list)

        elif self.mode == 'val':
            folds = self.folds_data_info_df
            folds = folds[folds.fold == fold_index]
            
            self.case_list = folds.case_name.values.tolist()
            self.data_file_list = folds.train_file_path.values.tolist()
            self.data_folder_list = folds.train_file_folder.values.tolist()
            self.data_type_list = folds.type.tolist()
            self.num_data = len(self.case_list)

        elif self.mode == 'test':
            self.case_list = [filename for filename in sorted(os.listdir(self.test_image_path)) if filename.endswith('.mhd')]
            self.data_file_list = [os.path.join(self.test_image_path, file_name) for file_name in self.case_list]
            self.num_data = len(self.data_file_list)

    def __getitem__(self, index):
        if self.fold_index is None and self.mode != 'test':
            raise ValueError

        label = None
        image_id = self.case_list[index]
        # ['train' 'val' 'test']
        if self.mode == 'test':
            image, source_size = self.compress_raw_dicom(os.path.join(self.test_image_path, self.data_file_list[index]), self.img_size,
                                                            data_format=self.data_format)
        else:
            try:
                start_point = None
                if self.scale != 'coarse':
                    if image_id not in self.key_start_points_dict.keys():
                        self.key_start_points_dict.update({image_id: self.get_key_mask_points(self.data_folder_list[index],
                                                                                              (self.img_size[0]*2,
                                                                                               self.img_size[1]*2))})
                    start_point = random.choice(self.key_start_points_dict[image_id])
                image, source_size = self.compress_raw_dicom(self.data_file_list[index], self.img_size, start_point,
                                                             data_format=self.data_type_list[index])
            except Exception as e:
                print('file: {} is not true dicom file, error: {}'.format(index, e))
                raise ValueError
            label = self.get_mask_multi(self.data_folder_list[index], source_size, self.img_size, start_point,
                                        data_format=self.data_type_list[index])

        # self.show_img_mask(image, label, 'pre_transform')
        if self.transform:
            if self.mode == 'test':
                sample = {'image': image}
            else:
                sample = {'image': image, 'mask': label}
            sample = self.transform(**sample)
            image = transform.ToTensor()(sample['image'])
            if not self.mode == 'test':
                # self.show_img_mask(sample['image'], sample['mask'], 'after_transform')
                label = torch.from_numpy(np.ascontiguousarray(sample['mask'])).long()

        if not self.mode == 'test':
            return image_id, image, label
        else:
            return image_id, image, source_size[0], source_size[1]

    def __len__(self):
        return self.num_data

    @staticmethod
    def get_key_mask_points(nii_data_path, img_size):
        def get_top_bottom_points(_mask, _img_size):
            height, width = _mask.shape[0], _mask.shape[1]
            point_y_top = np.nonzero(np.sum(_mask, axis=1))[0][0]
            point_x_top = (np.nonzero(_mask[point_y_top, :])[0][0] + np.nonzero(_mask[point_y_top, :])[0][-1]) // 2
            point_y_bot = np.nonzero(np.sum(_mask, axis=1))[0][-1]
            point_x_bot = (np.nonzero(_mask[point_y_bot, :])[0][0] + np.nonzero(_mask[point_y_bot, :])[0][-1]) // 2

            point_x_top = point_x_top - _img_size[1] // 2 if point_x_top - _img_size[1] // 2 > 0 else 0
            point_x_top = point_x_top if point_x_top + _img_size[1] < width else width - _img_size[1]
            point_x_bot = point_x_bot - _img_size[1] // 2 if point_x_bot - _img_size[1] // 2 > 0 else 0
            point_x_bot = point_x_bot if point_x_bot + _img_size[1] < width else width - _img_size[1]

            point_y_top = point_y_top - _img_size[0] // 2 if point_y_top - _img_size[0] // 2 > 0 else 0
            point_y_top = point_y_top if point_y_top + _img_size[0] < height else height - _img_size[0]
            point_y_bot = point_y_bot - _img_size[0] // 2 if point_y_bot - _img_size[0] // 2 > 0 else 0
            point_y_bot = point_y_bot if point_y_bot + _img_size[0] < height else height - _img_size[0]

            return [(point_x_top, point_y_top), (point_x_bot, point_y_bot)]

        assert os.path.exists(nii_data_path)
        key_points = []
        for file in os.listdir(nii_data_path):
            if file.lower().endswith('.nii.gz'):
                nii_data = nib.load(os.path.join(nii_data_path, file))
                _mask_array = np.asanyarray(nii_data.dataobj)
                if file not in special_data_list:
                    _mask_array = _mask_array[0].transpose(1, 0)
                else:
                    _mask_array = _mask_array[1].transpose(1, 0)
                _mask_array = np.flipud(_mask_array)
                # if file.endswith('1.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('2.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('3.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('4.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('5.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('6.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('7.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('8.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('9.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('10.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('11.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # elif file.endswith('12.nii.gz'):
                #     key_points.extend(get_top_bottom_points(_mask_array, img_size))
                # else:  # 假体
                #     top_p, bot_p = get_top_bottom_points(_mask_array, img_size)
                #     key_points.append(top_p)
                if file.lower().endswith('1.nii.gz') or file.lower().endswith('7.nii.gz'):
                    key_points.extend(get_top_bottom_points(_mask_array, img_size))
                elif file.lower().endswith('2.nii.gz') or file.lower().endswith('8.nii.gz'):
                    key_points.extend(get_top_bottom_points(_mask_array, img_size))
                elif file.lower().endswith('3.nii.gz') or file.lower().endswith('9.nii.gz'):
                    key_points.extend(get_top_bottom_points(_mask_array, img_size))
                elif file.lower().endswith('4.nii.gz') or file.lower().endswith('10.nii.gz'):
                    key_points.extend(get_top_bottom_points(_mask_array, img_size))
                elif file.lower().endswith('5.nii.gz') or file.lower().endswith('11.nii.gz'):
                    key_points.extend(get_top_bottom_points(_mask_array, img_size))
                else:
                    key_points.extend(get_top_bottom_points(_mask_array, img_size))
                    # top_p, bot_p = get_top_bottom_points(_mask_array, img_size)
                    # key_points.extend(get_top_bottom_points(_mask_array, img_size))
        # return random.choice(key_points)
        return key_points

    @staticmethod
    def get_mask_multi(nii_data_path, dcm_source_size, img_size, start_point=None, data_format='dcm'):
        assert os.path.exists(nii_data_path)
        mask_array = np.zeros(dcm_source_size)
        mask_array = mask_array[..., np.newaxis]
        mask_array = np.tile(mask_array, (1, 1, 6))
        for file in os.listdir(nii_data_path):
            if file.lower().endswith('.nii.gz'):
                nii_data = nib.load(os.path.join(nii_data_path, file))
                _mask_array = np.asanyarray(nii_data.dataobj)
                if file not in special_data_list:
                    if data_format == 'mhd':
                        _mask_array = _mask_array[:, :, 0].transpose(1, 0)
                    else:
                        _mask_array = _mask_array[:, :, 0].transpose(1, 0)
                else:
                    if data_format == 'mhd':
                        _mask_array = _mask_array[1].transpose(1, 0)
                    else:
                        _mask_array = _mask_array[:, :, 1].transpose(1, 0)


                if file.lower().endswith('_1.nii.gz') or file.lower().endswith('_7.nii.gz'):  # 股骨
                    mask_array[_mask_array != 0, 0] = 1
                elif file.lower().endswith('_2.nii.gz') or file.lower().endswith('_8.nii.gz'):  # 胫骨
                    mask_array[_mask_array != 0, 1] = 1
                elif file.lower().endswith('_3.nii.gz') or file.lower().endswith('_9.nii.gz'):  # 腓骨
                    mask_array[_mask_array != 0, 2] = 1
                elif file.lower().endswith('_4.nii.gz') or file.lower().endswith('_10.nii.gz'):  # 腓骨
                    mask_array[_mask_array != 0, 3] = 1
                elif file.lower().endswith('_5.nii.gz') or file.lower().endswith('_11.nii.gz'):  # 腓骨
                    mask_array[_mask_array != 0, 4] = 1
                elif file.lower().endswith('_6.nii.gz') or file.lower().endswith('_12.nii.gz'):  # 假体
                    mask_array[_mask_array != 0, 5] = 1
        if data_format == 'mhd':
            mask_array = np.flipud(mask_array)
        if start_point:
            mask_array = mask_array[start_point[1]: start_point[1] + img_size[1]*2,
                             start_point[0]: start_point[0] + img_size[0]*2, :]
        mask_array = cv2.resize(mask_array, img_size, interpolation=cv2.INTER_NEAREST).astype('uint8')
        return mask_array

    @staticmethod
    def show_img_mask(_normalized_img, mask_array, title):
        def convert_channel(original_data, label_data, value, channel, convert_value):
            for i in range(0, 3):
                if i == channel:
                    original_data[label_data == value, channel] = convert_value
        normalized_img = np.asarray(_normalized_img).copy()

        convert_channel(normalized_img, mask_array[:, :, 0], 1, 0, 255)
        convert_channel(normalized_img, mask_array[:, :, 0], 1, 1, 255)
        convert_channel(normalized_img, mask_array[:, :, 0], 1, 2, 0)
        convert_channel(normalized_img, mask_array[:, :, 1], 1, 0, 255)
        convert_channel(normalized_img, mask_array[:, :, 1], 1, 1, 0)
        convert_channel(normalized_img, mask_array[:, :, 1], 1, 2, 0)
        convert_channel(normalized_img, mask_array[:, :, 2], 1, 0, 0)
        convert_channel(normalized_img, mask_array[:, :, 2], 1, 1, 255)
        convert_channel(normalized_img, mask_array[:, :, 2], 1, 2, 0)
        convert_channel(normalized_img, mask_array[:, :, 3], 1, 0, 0)
        convert_channel(normalized_img, mask_array[:, :, 3], 1, 1, 0)
        convert_channel(normalized_img, mask_array[:, :, 3], 1, 2, 255)
        convert_channel(normalized_img, mask_array[:, :, 4], 1, 0, 255)
        convert_channel(normalized_img, mask_array[:, :, 4], 1, 1, 0)
        convert_channel(normalized_img, mask_array[:, :, 4], 1, 2, 255)
        convert_channel(normalized_img, mask_array[:, :, 5], 1, 0, 0)
        convert_channel(normalized_img, mask_array[:, :, 5], 1, 1, 255)
        convert_channel(normalized_img, mask_array[:, :, 5], 1, 2, 255)



        normalized_img = cv2.resize(normalized_img, (normalized_img.shape[1]//6, normalized_img.shape[0]//6))
        from matplotlib import pyplot as plt
        plt.figure(title)
        plt.imshow(normalized_img)
        plt.show()

    @staticmethod
    def compress_raw_dicom(file_path, img_size, start_point=None, data_format='dcm'):

        def get_img_max_min(img_array, clip_lower=5, clip_upper=95):
            hypo_min_v = np.percentile(img_array, clip_lower)
            hypo_max_v = np.percentile(img_array, clip_upper)
            return hypo_min_v, hypo_max_v

        if data_format != 'mhd':
            dcm_pixel_array = pydicom.dcmread(file_path).pixel_array
        else:
            img=cio.read_image(file_path, dtype=np.int16).to_numpy()
            dcm_pixel_array = cio.read_image(file_path, dtype=np.int16).to_numpy()[0, :, :]
            dcm_pixel_array = np.flipud(dcm_pixel_array)

        dcm_height, dcm_width = dcm_pixel_array.shape
        hypo_pixel_min, hypo_pixel_max = get_img_max_min(dcm_pixel_array)

        dcm_pixel_array_fixed = np.clip(dcm_pixel_array, hypo_pixel_min, hypo_pixel_max).astype('float64')
        normalized_img = (dcm_pixel_array_fixed - hypo_pixel_min) / (hypo_pixel_max - hypo_pixel_min)

        # opencv work
        normalized_img = (normalized_img * 255.0 + 0.5).clip(0, 255).astype('uint8')

        if start_point:
            normalized_img = normalized_img[start_point[1]: start_point[1] + img_size[1]*2,
                             start_point[0]: start_point[0] + img_size[0]*2]
        normalized_img = cv2.resize(normalized_img, img_size, interpolation=cv2.INTER_LINEAR)

        # more preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        normalized_img = clahe.apply(normalized_img)
        normalized_img = np.clip(normalized_img, 0, 255).astype('uint8')
        # end

        normalized_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)

        return normalized_img, [dcm_height, dcm_width]


def get_k_fold_cross_validation_df(train_image_path, data_path_extra, n_folds):
    assert os.path.exists(train_image_path)

    train_dict = {case_name: os.path.join(train_image_path, case_name)
                  for case_name in os.listdir(train_image_path)}
    train_list = list(train_dict.keys())
    random.shuffle(train_list)
    length = math.ceil(len(train_list) / n_folds)

    train_file_folder_list = [train_dict[case] for case in train_list]
    train_file_path_list = [os.path.join(train_dict[case], [file_name for file_name in os.listdir(train_dict[case])
                                                            if file_name.lower().endswith('.mhd') or
                                                            file_name.lower().endswith('.dcm')][0])
                            for case in train_list]
    fold_list = [index // length for index, case in enumerate(train_list)]
    file_type_list = ['mhd' if str(file_name).endswith('mhd') else 'dcm' for file_name in train_file_path_list]
    _df = pd.DataFrame({'case_name': train_list, 'train_file_folder': train_file_folder_list,
                       'train_file_path': train_file_path_list, 'fold': fold_list, 'type': file_type_list})

    if data_path_extra:
        for _extra_path in data_path_extra:
            extra_df = get_k_fold_cross_validation_df(_extra_path, [], n_folds)
            _df = pd.concat([_df, extra_df])
    return _df


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import albumentations as albu
    import random
    from tqdm import tqdm

    # data_folder_ = '/data/jbwu_data/xiazhi_data/clean_data/source_data/'
    data_folder_ = '/home/htwang/data/source_data_test'
    extra_data = []
    # data_folder_ = '/data/jbwu_data/xiazhi_data/clean_data/source_data_1000_sub_1'

    # extra_data = ['/data/jbwu_data/xiazhi_data/clean_data/source_data_1000_sub_1']
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    img_size_ = (768, 768)
    data_df = get_k_fold_cross_validation_df(data_folder_, extra_data, 5)
    # print(data_df[:10])

    train_transform = albu.load('./transforms/train_transforms_complex_768_box.json')
    seg_dataset = SegDataset(img_size=img_size_, mode='train', data_transform=train_transform,
                             fold_index=0, folds_data_info_df=data_df, scale='coarse')
    # not use sampler
    data_loader = DataLoader(dataset=seg_dataset, batch_size=8, num_workers=0)
    for batch_idx, (_, imgs, labels) in tqdm(enumerate(data_loader)):
        # pass
        if batch_idx >= 1:
            break