import os
import pickle
from PIL import Image


def image_preprocess(folder_path, size):
    li_folder= folder_path.split('/')
    resized_folder = li_folder[-1]+'_resized_' + str(size)
    resized_folder_path = '/'.join(li_folder[:-1])+ '/' +resized_folder
    print(resized_folder_path)
    if not os.path.exists(resized_folder_path):
        os.makedirs(resized_folder_path)
    file_list = os.listdir(folder_path)
    print('No of images in your folder: '+str(len(file_list)))
    cnt=0
    for image in file_list:
        if cnt%100==0:
            print("Images processed: "+str(cnt))
        try:
            im = Image.open(folder_path + '/' + image)
        except:
            print("File not processed:"+image)
            continue
        im = im.resize([size, size], Image.ANTIALIAS)
        im.save(resized_folder_path + '/' + image)
        cnt+=1
