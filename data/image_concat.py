
from PIL import Image

import os
def join(png1,png2,result_dir, flag='horizontal'):
    """
    :param png1: path
    :param png2: path
    :param flag: horizontal or vertical
    :return:
    """
    img1, img2 = Image.open(png1), Image.open(png2)
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        joint = Image.new('RGB', (size1[0]+size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(result_dir)
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1]+size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(result_dir)


if __name__ == '__main__':

    haze_dir="/home/qiaoyingxv/datasets/coal_datasets_0608/syn_haze_resize_new"
    gt_dir="/home/qiaoyingxv/datasets/coal_datasets_0608/syn_gt_resize_new"
    out_dir="/home/qiaoyingxv/datasets/coal_datasets_0608/train"
    haze_paths=sorted(os.listdir(haze_dir))
    gt_paths = sorted(os.listdir(gt_dir))
    for img_name in haze_paths:
        haze_img_dir = os.path.join(haze_dir, img_name)
        gt_img_dir = os.path.join(gt_dir,img_name)
        out_img_dir = os.path.join(out_dir,img_name)

        join(haze_img_dir, gt_img_dir,out_img_dir)
        #join(png, png, flag='vertical')
