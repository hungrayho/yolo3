import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import json

ia.seed(1)

with open('coco_original.json') as json_file:  
    data = json.load(json_file)

#used to determine range of iterations
img_count = len(data['images']) + 1
img_count_init = len(data['images'])
anno_count = len(data['annotations']) + 1
anno_count_init = len(data['annotations'])

#number of augmentations per image
aug_no = 4

#directory path for images
img_dir_path = 'images/'

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        #
        # Apply the following augmenters to most images.
        #
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images

        # crop some of the images by 0-10% of their height/width
        sometimes(iaa.Crop(percent=(0, 0.1))),

        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),

        #
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0, 5),
            [
                # Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image, but do
                # not replace all superpixels with their average, only
                # some of them (p_replace).
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Same as sharpen, but for an embossing effect.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

                # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                # Invert each image's chanell with 5% probability.
                # This sets each pixel value v to 255-v.
                iaa.Invert(0.05, per_channel=True), # invert color channels

                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),

                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # Improve or worsen the contrast of images.
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random
                # strengths).
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # In some images distort local areas with varying strength.
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    # do all of the above augmentations in random order
    random_order=True
)

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

with open('coco_heavy.json', 'w') as outfile:
    json.dump(data, outfile)