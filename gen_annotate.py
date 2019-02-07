# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:24:09 2018

@author: Chris
"""
#%%
#script to convert coco.json to coco.txt for yolo
import json

with open('coco.json') as json_file:
    data = json.load(json_file)

f= open("avg_train.txt","w+")

image_id = 1
f.write("avg/" + data['images'][0]['file_name'])

for i in range(0, len(data['annotations'])):
    if data['annotations'][i]['image_id'] == image_id:
        f.write(" " + str(data['annotations'][i]['bbox'][0]) + ","
               + str(data['annotations'][i]['bbox'][1]) + ","
               + str(data['annotations'][i]['bbox'][0] + data['annotations'][i]['bbox'][2]) + ","
               + str(data['annotations'][i]['bbox'][1] + data['annotations'][i]['bbox'][3]) + ","
               + str(data['annotations'][i]['category_id'] - 1))
    else:
        image_id = data['annotations'][i]['image_id']
        f.write("\ravg/" + data['images'][image_id - 1]['file_name'])
        f.write(" " + str(data['annotations'][i]['bbox'][0]) + ","
               + str(data['annotations'][i]['bbox'][1]) + ","
               + str(data['annotations'][i]['bbox'][0] + data['annotations'][i]['bbox'][2]) + ","
               + str(data['annotations'][i]['bbox'][1] + data['annotations'][i]['bbox'][3]) + ","
               + str(data['annotations'][i]['category_id'] - 1))

f.close()