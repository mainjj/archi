import cv2
import pandas 
import os
import csv
import argparse

def input_key(val, csv_path):        
    key = cv2.waitKey(0) #input key
    f = open(csv_path,'a', newline='') # csv open
    wr = csv.writer(f) 
        
    if key == ord('p'): # is person 
        wr.writerow([1,*val[5:]]) 
        f.close()
        return True
    
    elif key == ord('n'): # is not person
        wr.writerow([0,*val[5:]])
        f.close()
        return True
    
    else: #wrong input
        print("Wrong key input\tplease input keys [p, n]\n\t- p: BBox is person\n\t- n: BBox is not person")
        return False



# argument
#################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', type=str, help='data.csv file path')
parser.add_argument('-sp','--savepath', default='./', help='save path, default: ./')
opt = parser.parse_args()
#################################################################################################################

data_path, save_path = opt.data, opt.savepath
csv_path = os.path.join(save_path, 'train_data.csv')

# get data
data = pandas.read_csv(data_path,header=None)

# window setting
cv2.namedWindow('IMAGE', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow(winname='IMAGE', width=640, height=640)

# loop
for index in data.index: 
    val = data.loc[index]
    img_path, xmin, ymin, xmax, ymax = val[0],val[1],val[2],val[3],val[4] # image, bbox parsing
    
    img = cv2.imread(img_path) # read image
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2) #draw bbox
    cv2.imshow('IMAGE',img) # image show
    while not input_key(val, csv_path): # wait key-input
        continue            

cv2.destroyAllWindows()

