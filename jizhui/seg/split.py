import datetime
import os
import md.image3d.python.image3d_io as cio
import md.image3d.python.image3d_tools as ctools
import numpy as np
from md import Image3d
import nibabel as nib

print('Hello World!')

print('Time is ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %A'))


def main():
    path = '/home/htwang/data/test_split'
    files = os.listdir(path)
    for file in files:
        if file.endswith('.nii.gz'):
            prefix = file.split('.nii.gz')[0]
            f = cio.read_image(os.path.join(path, file))
            temp=f.to_numpy()
            print(temp)
            mask = f.to_numpy()[0]
            mask_frame = f.frame().deep_copy()
            for num in range(1, 13):
                mask_3d = Image3d()
                temp = (mask == num) * num
                print(temp)
                img = temp.astype(np.uint8)
                # img = (temp * 255).astype(np.uint8)
                mask_3d.from_numpy(np.expand_dims(img, 0))
                mask_3d.set_frame(mask_frame)
                cio.write_image(mask_3d, '/home/htwang/data/test_split/'+prefix + '_' + str(num) + '.nii.gz')
            os.remove(path+'/'+file)


            # print(f)
            # ctools.convert_labels_outside_to_zero(f,0,11)


if __name__ == '__main__':
    main()
