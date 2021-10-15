import os
import argparse
import selectivesearch
import json
from tqdm import tqdm
from skimage import io as skio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Generate result of selective search for SOCO')
    parser.add_argument('--label-path', default='/mnt/lustre/share/images/meta/train.txt')
    parser.add_argument('--out-path', default='label/imagenet_ssearch_train.json')
    parser.add_argument('--image-path', default='/mnt/lustre/share/images/train/')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    args = parser.parse_args()

    root_path = '/mnt/lustre/share/images/'
    label_file = args.label_path
    out_path = args.out_path
    image_path = args.image_path
    idx_start = args.start

    idx_end = args.end


    out_path = out_path.split(".")[0] + str(idx_start) + "_" + str(idx_end) + ".json"

    f_read = open(label_file)
    f_write = open(out_path, 'w')

    lines = f_read.readlines()
    if idx_end < 0:
        idx_end = len(lines)

    lines = lines[idx_start:idx_end]

    for line in tqdm(lines):
        filename = line.split(' ')[0]
        a_image_path = os.path.join(image_path, filename)
        image = skio.imread(a_image_path)
        if len(image.shape) != 3:
            print(filename + " :" + str(image.shape))
            image = np.expand_dims(image, 2).repeat(3, axis=2)

        img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=10)
        out_dict = {'filename': filename, 'regions': regions}
        f_write.write(json.dumps(out_dict) + "\n")

if __name__ == '__main__':
    main()

