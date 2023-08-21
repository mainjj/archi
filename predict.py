import os
import cv2
import csv
import time
import pickle
import argparse
import torch
import sys

sys.path.append('./yolov7')

from  yolov7.hubconf import custom
from tools.model_load import attempt_load,Ensemble
from tools.img_process import parse_boxes, img_preprocessing, draw_detections

# argument
#################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-r','--region-proposal', type=str, help='model.pt path(s) for region proposal')
parser.add_argument('-c','--clf-weights', type=str, help='txt file path for classification')
# parser.add_argument('-c','--clf-weights', nargs='+', type=str, help='model.pt path(s) for classification, ex) -c ./model1.pt ./model2.pt')
parser.add_argument('-l','--logistic-regressor', type=str, help='model.pickle path(s) for logistic regression')
parser.add_argument('-s','--source', type=str, help='source')
parser.add_argument('-p','--project', default='./runs', help='save results to project/name, default: ./runs')
parser.add_argument('-n','--name', default='exp', help='save results to project/name')
parser.add_argument('--task', default='predict', help='[predict or save], "save" to labeling, default: predict')
parser.add_argument('--show-all', action='store_true', help='plot all detection result')
opt = parser.parse_args()
print(opt)
#################################################################################################################
# setup
region_proposal_path = opt.region_proposal # model.pt path for region proposal
# clf_weights = opt.clf_weights # model(s).pt path(s) for classification
with open(opt.clf_weights,'r') as c_f:
    clf_weights = c_f.read().split()
logistic_regressor_path = opt.logistic_regressor # model.pickle path for logistic regression

source_dir = os.path.join(opt.source)
save_dir = os.path.join(opt.project,opt.name) #save dir path
if not os.path.exists(save_dir): 
    os.mkdir(save_dir) # make save dir


image_list = os.listdir(source_dir) # get image names
SHOW_ALL = opt.show_all # plot not-person bbox OR not
#check task
if opt.task != 'save' and opt.task != 'predict': raise Exception(f'unknown argument, task: {opt.task}')
is_save = True if opt.task == 'save' else False
print(f"Task is {opt.task}")
#################################################################################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

region_proposal = custom(path_or_model=region_proposal_path)
# region_proposal = torch.hub.load('WongKinYiu/yolov7','custom',region_proposal_path)  #region proposal (yolov7) load by torch.hub (it will be change)
region_proposal.conf = 0.01  # NMS confidence threshold                                                   
# region_proposal.agnostic = True  # NMS class-agnostic
region_proposal.eval()
#region_proposal.cpu()


# classification models load
clf_models = attempt_load(clf_weights,Ensemble(), map_location=device)

# Logistic Regression model load
if not is_save:
    loaded_model = pickle.load(open(logistic_regressor_path,'rb'))
    print(f"Logistic Regressor load: {logistic_regressor_path}")

total_time = 0
#iter per image
for img_name in image_list:
    start_time = time.time() #for check elapstime
    #img read
    f = cv2.imread(os.path.join(source_dir,img_name))
    #Do region proposal
    with torch.no_grad():
        results = region_proposal([f])
    boxes= parse_boxes(results, conf_thres=0.01) #Boxes parsing
    # detections = draw_detections(boxes, f.copy())  #plot region

    draw_img = f.copy() #for save img
    for i, box in enumerate(boxes): #select one Region
        print(i,'th region /',len(boxes))
        region = draw_img[box[1]:box[3], box[0]:box[2], :]  # region parsing
        region = img_preprocessing(region,device) # preprocess for inference
        # Region Classification
        with torch.no_grad():
            pred = clf_models(region)
        print('clf_pred: ',pred)

        if is_save:
            csv_f = open(os.path.join(save_dir,'save_for_labeling.csv'),'a',newline='')
            csv_wr = csv.writer(csv_f)
            csv_wr.writerow((os.path.join(source_dir,img_name),*box,*pred))
            csv_f.close()
            print(f'({img_name}) write csv row')
            continue


        # Logistic Regression predict
        log_pred = loaded_model.predict([pred])
        print('is person |',log_pred,'\n')

        # Drawing
        if log_pred[0] == 1: #person plot
            draw_img = draw_detections([box],draw_img, f' ')
        
        elif log_pred[0] == 0 and SHOW_ALL: #not-person  plot
            draw_img = draw_detections([box],draw_img, f' ',(0,0,0))
            # draw_img = draw_detections([box],draw_img, f' ')

    if is_save: 
        total_time += time.time()-start_time
        continue
    # save img
    cv2.imwrite(os.path.join(save_dir,img_name),draw_img)
    elapsedtime = time.time()-start_time

    print(f'save in {save_dir} ({round(elapsedtime,4)}s)')
    total_time += elapsedtime

print(f'DONE\nwall time: {round(total_time,2)}s\naverage time: {round(total_time/len(image_list),4)}s FPS: {1/(total_time/len(image_list))}')

sys.path.remove('./yolov7')