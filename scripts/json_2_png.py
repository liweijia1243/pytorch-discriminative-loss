import json
import numpy as np
import argparse
import glob
import cv2
import matplotlib.pyplot as plt

def getfiles(json_path, save_path):
    print(json_path)
    file_list = glob.glob(json_path+"/*.json")
    print(file_list)
    for file in file_list:
        with open(file,"r") as f:
            json_root = json.load(f)
            #print(json_root)
            for img in json_root:
                saveimages(img, save_path)
    pass


def saveimages(img_json : dict, save_path):
    print("get one image label file name:{0}".format(img_json["image"]))
    #img = np.ndarray((512,512),dtype=np.uint8)
    img = np.zeros((512,512),dtype=np.uint8)
    lines = []
    for cluster in img_json["cluster"]:
        polyline=[]
        for point in cluster["points"]:
            polyline.append(point)
        lines.append(polyline)
    #draw img
    img_name = img_json["image"].split("/")[-1]
    #print(img_name)
    cluster_id = 1
    cluster_count = len(lines)
    for line in lines:
        line = np.array(line, dtype=np.double).reshape(-1,1,2)
        line = np.array(line/100*512,dtype=np.int32)
        cv2.polylines(img,[line],False,255*cluster_id//cluster_count,3)
        #print(255*cluster_id//cluster_count)
        cluster_id =cluster_id+1
    cv2.imwrite(save_path+"/"+img_name, img)
    # plt.imshow(img)
    # plt.show()
    pass

def main(args):
    getfiles(args.json_path, args.save_path)
    pass

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--json_path", nargs='?', default="./")
    parse.add_argument("-s", "--save_path", nargs='?', default="./")
    args = parse.parse_args()
    main(args)