import os
from tqdm import tqdm
from PIL import Image

def _resize(source_dir, target_dir):
    print(f'Start resizing {source_dir} ')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for i in tqdm(os.listdir(source_dir)):
        filename = os.path.basename(i)
        try:
            image = Image.open(os.path.join(source_dir, i)).convert('RGB')
            #if len(image.shape) == 3 and image.shape[-1] == 3:
            W ,H  = image.size

            if W>H :
                ratio = W / H
                H = 512
                W = int(ratio * H)
                W= int((W//32)* 32)
            else:
                ratio = H / W
                W =512
                H = int(ratio * W)
                H = int((H//32)*32)
            image1 = image.resize((W,H),Image.ANTIALIAS)
            #image1=center_crop(image1,(768,768))
            image1.save(os.path.join(target_dir, filename))
        except:
            continue

source_dir="/home/qiaoyingxv/datasets/enhance_dehazing/URHI"
target_dir ="/home/qiaoyingxv/datasets/enhance_dehazing/URHI_resized"
_resize(source_dir, target_dir)