# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:19:44 2018

@author: Chris
"""
#%%
import imgaug as ia
from imgaug import seed
from imgaug import augmenters
import numpy as np
from PIL import Image
import json

seed(1)

with open('images/bboxes_coco/coco.json') as json_file:
    data = json.load(json_file)

#used to determine range of iterations
img_count = len(data['images']) + 1
img_count_init = len(data['images'])
anno_count = len(data['annotations']) + 1
anno_count_init = len(data['annotations'])

#number of augmentations per image
aug_no = 4

#directory path for images
img_dir_path = 'images/pikachu/'

#%%
seq = augmenters.Sequential([
    augmenters.Fliplr(0.5), # horizontal flips
    augmenters.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    augmenters.Sometimes(0.5,
        augmenters.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    augmenters.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    augmenters.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    augmenters.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

#dictionary to tag bounding boxes to respective image
bbs_dict = {k: [] for k in range(0, img_count)}

for i in range(0, anno_count_init):
    bbs_dict[data['annotations'][i]['image_id'] - 1].append(i)

#iterates over all images logged in coco.json
for i in range(0, img_count_init):
    try:
        #converts image to np array then duplicates aug_no times. Each instance will be run through the imgaug sequence
        image = np.array(
        [np.asarray(Image.open(img_dir_path + data['images'][i]['file_name'])) for _ in range(aug_no)],
        dtype=np.uint8)

        #from dictionary pulls all related bboxes, then again duplicate aug_no times
        bbs = []
        img_h = data['images'][i]['height']
        for b in range(0, len(bbs_dict[i])):
            bbox = data['annotations'][bbs_dict[i][b]]['bbox']
            bbs.append(ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0]+bbox[2], y2=bbox[1]+bbox[3]))
        bbs_oi = np.array([ia.BoundingBoxesOnImage(bbs, shape=image[0].shape) for _ in range(aug_no)])

        #applies deterministic sequence so both image and bboxes are transformed in the same way
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images(image)
        bbs_aug = seq_det.augment_bounding_boxes(bbs_oi)

        #iterates over each augmented image, saves image, then appends info on image and bboxes to json
        for j in range (0, len(image_aug)):
            im = Image.fromarray(image_aug[j])
            im.save(img_dir_path + data['images'][i]['file_name'].split(".")[0] + "_" + str(j) + ".jpg")
            data['images'].append({'id': img_count,
                                   'file_name': data['images'][i]['file_name'].split(".")[0] + "_" + str(j) + ".jpg",
                                   'width': data['images'][i]['width'],
                                   'height': data['images'][i]['height']})
            curr_bbs = bbs_aug[j].remove_out_of_image().cut_out_of_image().to_xyxy_array()
            for b in range(0, len(bbs_dict[i])):
                curr_anno = bbs_dict[i][b]
                data['annotations'].append({'segmentation': [int(curr_bbs[b][0]), int(curr_bbs[b][1]), int(curr_bbs[b][0]), int(curr_bbs[b][3]), int(curr_bbs[b][2]), int(curr_bbs[b][3]), int(curr_bbs[b][2]), int(curr_bbs[b][1])],
                                           'area': int(curr_bbs[b][2]-curr_bbs[b][0])*int(curr_bbs[b][3]-curr_bbs[b][1]),
                                           'iscrowd': data['annotations'][curr_anno]['iscrowd'],
                                           'ignore': data['annotations'][curr_anno]['ignore'],
                                           'image_id': img_count,
                                           'bbox': [int(curr_bbs[b][0]), int(curr_bbs[b][1]), int(curr_bbs[b][2]-curr_bbs[b][0]), int(curr_bbs[b][3]-curr_bbs[b][1])],
                                           'category_id': data['annotations'][curr_anno]['category_id'],
                                           'id': anno_count})
                anno_count += 1
            img_count += 1
        print('Processed image %d' % i)
    except:
        print('Error occured when processing image %d' % i)

with open('coco_new.json', 'w') as outfile:
    json.dump(data, outfile)
 
