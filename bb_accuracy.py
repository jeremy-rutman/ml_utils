__author__ = 'jeremy'

import msgpack
import requests

from . import imutils

__author__ = 'jeremy'

import msgpack
import requests
import cv2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pdb
import copy
import pandas as pd
import os

from . import Utils
from . import bb_results
from . import read_various_training_formats


def compare_bb_dicts(gt_list,guess_list,dict_format={'bbox':'bbox','object':'object','confidence':'confidence'},iou_threshold=0.2):
    '''
    given 2 dicts of bbs - find bb in dict2 having most overlap for each bb in dict1 (assuming thats the gt)
    for each gt:
    find overlapping guesses where ovrerlap iou > 0.5
    take most confident guess
    any other overlaps are counted as false positive
    If there's no overlapping bb or the cat. is wrong thats a false negative.
    iou counts for average even if category is wrong (check this against standards...!)
    iou=0 if no guess box overlaps a gt box.
    extra detections with no iou>0.5 count as false pos, and contribute iou=0 to average
    mAP computed by averaging precision values on p-r curve at steps [0,0.1,...,1]
     To be more precise, we consider a slightly corrected PR curve, where for each curve point (p, r), if there is a
     different curve point (p', r') such that p' > p and r' >= r,
     we replace p with maximum p' of those points.
    there are no true negatives here to speak of

    see https://stackoverflow.com/questions/36274638/map-metric-in-object-detection-and-computer-vision
    and http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

    Detections output by a method were assigned to ground
    truth objects satisfying the overlap criterion in order ranked by the (decreasing) confidence output. Multiple detections
    of the same object in an image were considered false detections e.g. 5 detections of a single object counted as 1 correct
    detection and 4 false detections  it was the responsibility of
    the participants system to filter multiple detections from its output


    What is still unclear to me is what is done with those GT boxes that are never detected (even if the confidence is 0)
    This means that there are certain recall values that the precision-recall curve will never reach, and this makes the
    average precision computation above undefined.

    I take GT's not detected as false negative

    That is not the only thing that is unclear. Consider a case where there are two predicted boxes (P1, P2) and two
    ground-truth boxes (T1, T2),
    where P2 has higher confidence than P1. Both P1 and P2 overlap T1. Since P2 has the higher confidence, it is clear
    that P2 should be considered
    the match for T1. What is not given is if P1 also has some IOU overlap with T2, but lower than the IOU with T1, should
    P1 be given a second chance to try to match itself to T2,
    or should it not

    I am allowing this in the code below. only the matching (highest conf) prediction is marked as 'already matched'

    case three:
    Say that P2 also overlaps T2 and has higher IOU than with T1, then it seems that P2 should be matched with T2 and not T1
    this isn't taken care of in code below...

    case 4 - two different guesses have same IOU with same T and same confidence

    :param dict1:ground truth list [{ 'object': 'bag', 'bbox': [454, 306, 512, 360]},...,]
                guess in same form but with confidence (gt can also have confidence 1.0)
            bbox here is xywh , aka x1 y1 w h , coords are 'regular' image coords
            (origin is top left, positive x goes right and pos y goes down)
    :param dict2:guess in 'api form'
    :param dict_format - this lets you use dicts in different formats, just substitute whatver term is used into the
    dict

    e.g.         if the dict uses 'x_y_w_h' instead of 'bbox' and 'annotations' instead of 'data' and 'label'
     instead
     of 'object'
    then dict_format = {'data':'annotations', 'bbox':'w_y_w_h','object:'label'}
    :return:  n_true_positive, n_false_neg, n_false_pos, avg_iou
    '''

    gt_data=gt_list
    guess_data=guess_list

    true_pos = 0
    false_pos = 0 #aka lonely hypothesis
    false_neg = 0  #aka lonely ground truth
    #there are no true negatives here to speak of
    iou_tot = 0
    n_detections = 0  #this includes matching detections both above and below threshold.
    detections_over_threshold = 0
    obj_kw = dict_format['object']
    bb_kw = dict_format['bbox']
    conf_kw = dict_format['confidence']

    #take care of degenerate cases - no gts or no guesses or both
    if len(gt_data) == 0 and len(guess_data) == 0:
        return {'tp':0,'fp':0,'fn':0,'iou_avg':0,'iou_accumulator':0,'n_ious':0}
    if len(gt_data) == 0 : #all guesses are fp
        return {'tp':0,'fp':len(guess_data),'fn':0,'iou_avg':0,'iou_accumulator':0,'n_ious':len(guess_data)}
    if len(guess_data) == 0 : #all gt are fn
        return {'tp':0,'fp':0,'fn':len(gt_data),'iou_avg':0,'iou_accumulator':0,'n_ious':len(guess_data)}



# first pass, determine ious of all overlapping bbs
    iou_table = np.zeros([len(gt_data),len(guess_data)])
    confidences = np.zeros(len(guess_data))
    for j in range(len(guess_data)):
        confidences[j] = guess_data[j][conf_kw]
        for i in range(len(gt_data)):
            if gt_data[i][obj_kw]!=guess_data[j][obj_kw]: #mismatched objects types
                continue
            iou = Utils.intersectionOverUnion(gt_data[i][bb_kw],guess_data[j][bb_kw])
            iou_table[i,j] = iou
    print('confidences:'+str(confidences))

    #resolve conflicting entries - use best match to T for given P1
    #if P1 and P2 both have iou>0 for T, use the one with higher conf
    #loser now has its highest match out of the running but rest still in

    ious_copy = copy.copy(iou_table)
    conflict = detect_conflict(ious_copy)
    while(conflict is not None):
        print('iou'+str(ious_copy))
        print('conflict:'+str(conflict))
        ious_copy = resolve_conflict(ious_copy,conflict,confidences)
        conflict = detect_conflict(ious_copy)
    print('no more conflict, iou {}'.format(ious_copy))

    #now keep only highest iou in given column to avoid multiple matches
    ious_one_per_column = np.zeros_like(ious_copy)
    for col in range(ious_copy.shape[1]):
        new_col = np.where(ious_copy[:,col]==np.max(ious_copy[:,col]),np.max(ious_copy[:,col]),0)
        ious_one_per_column[:,col] = new_col
    print('one per column:{}'.format(ious_one_per_column))

    # #second pass, match bb w. highest conf to box w highest ious
    eps = 10**-10
    guess_matched_with_gt = np.zeros(iou_table.shape[1])
    for row in range(ious_copy.shape[0]):
        above_thresh = np.where(ious_one_per_column[row,:]>0,1,0)
        n_above = np.sum(above_thresh)
        assert n_above<2, 'More than one entry in resolved iou matrix was above thresh!'
        if n_above == 0:
            false_neg += 1  # best guess (if any)  has iou < thresh
            print('nothing above threshold')
            if np.sum(ious_one_per_column[row,:]) < eps: #no ious in this row
                print('zero sum row ')
                continue
            iou_tot += np.max(ious_one_per_column[row,:])  #this will include the best lower-than-thresh detection, if any
            index = np.argmax(ious_one_per_column[row,:])
            guess_matched_with_gt[index] = 1
            n_detections += 1
        elif n_above == 1:
            detections_over_threshold += 1
            n_detections += 1
            true_pos += 1
            iou_tot += np.max(ious_one_per_column[row,:])
            index = np.argmax(ious_one_per_column[row,:])
            guess_matched_with_gt[index] = 1
        else: #this should not happen afer conflict resolution done above
            logging.warning('multiple matches found for object {}'.format(row))
        print('tp {} fn {} avg_iou {} tot_iou {}'.format(true_pos,false_neg,iou_tot/n_detections,iou_tot))

    #check for extra guess detections
    print('guess_matched with gt:'+str(guess_matched_with_gt))
    for col in range(len(guess_matched_with_gt)):
        if guess_matched_with_gt[col] == 0 :
            print('guess {} is a false pos'.format(col))
            false_pos += 1
            #should false positives affect IOU ? if so put that here...
            n_detections += 1 #yes, false pos adds 0 to iou running sum and decreases avg
    if n_detections == 0:
        iou_avg = 0
        logging.warning('no detections above thresh !')
    else:
        iou_avg = iou_tot/n_detections
    print('final tp {} fp {} fn {} avg_iou {}'.format(true_pos,false_pos,false_neg,iou_avg))
    return {'tp':true_pos,'fp':false_pos,'fn':false_neg,'iou_avg':iou_avg,'iou_accumulator':iou_tot,'n_ious':n_detections}

def resolve_conflict(ious_over_thresh_copy,conflict,confidences):
    if confidences[conflict[0]]>confidences[conflict[1]]:
        ious_over_thresh_copy[conflict[2],conflict[1]] = 0
    else:
        ious_over_thresh_copy[conflict[2],conflict[0]] = 0
    return ious_over_thresh_copy

def compare_bb_dicts_class_by_class(gt_dict,guess_dict,
                                    dict_format={'data':'data','bbox':'bbox','object':'object','confidence':'confidence'},
                                    iou_threshold=0.2,visual_output=True,img_arr=None):
    classes = get_classes_in_dicts([gt_dict,guess_dict])
    for cl in classes:
        gts=[]
        guesses=[]
        for annotation in gt_dict[dict_format['data']]:
            if annotation[dict_format['object']]==cl:
                gts.append(annotation)
        for annotation in guess_dict[dict_format['data']]:
            if annotation[dict_format['object']]==cl:
                guesses.append(annotation)
        print('class {} gts {}\nguesses {}'.format(cl,gts,guesses))
        if gts == []:
            print('no gt for {}'.format(cl))
            #count guesses for this class - taken care of in compare_bb_dicts
        if guesses == []:
            print('no guesses for {}'.format(cl))
            #count gts for this class - taken care of in compare_bb_dicts
        if visual_output:
            display_dicts(copy.copy(img_arr),gts,guesses,dict_format=dict_format)
        results = compare_bb_dicts(gts,guesses,dict_format=dict_format)
        print('results for {}: {}'.format(cl,results))

def display_dicts(img_arr,gts,guesses,dict_format = {'data':'data','bbox':'bbox','object':'object','confidence':'confidence'}):
    if img_arr is None:
        print('got none for '+img_arr)
        return
    for gt_obj in gts:
        img_arr = imutils.bb_with_text(img_arr,gt_obj[dict_format['bbox']],gt_obj[dict_format['object']],boxcolor=[255,0,0])
    for obj in guesses:
        img_arr = imutils.bb_with_text(img_arr,obj[dict_format['bbox']],obj[dict_format['object']]+' '+str(obj['confidence']),boxcolor=[0,255,0])
    cv2.imshow('img',img_arr)
    cv2.waitKey(0)


def detect_conflict(iou_matrix,iou_threshold = 0.2):
    '''
    test iou mat for row with two above-threshold entries P1,P2  - this means two bbs
    are competing for a single ground truth T1 and the one with higher conf. should win - as long
    as it has no higher-iou matches with some other ground truth T2. if it does then match P with that T
    so comparison is  between highest-iou matches for a given P
    :param argsorted_matrix:
    :param iou_threshold:
    :return:
    '''
  #  pdb.set_trace()
    for col in range(iou_matrix.shape[1]):
        if np.all(iou_matrix[:,col]<iou_threshold):
 #           print('entire col1 {} is below thresh, no conf'.format(col))
            continue
        index_of_highest_in_col = np.argmax(iou_matrix[:,col])
        for col2 in range(col+1,iou_matrix.shape[1]):
            if np.all(iou_matrix[:,col2]<iou_threshold):
  #              print('entire col2 {} is below thresh, no conf'.format(col2))
                continue
            index_of_highest_in_col2 = np.argmax(iou_matrix[:,col2])
            if index_of_highest_in_col == index_of_highest_in_col2:
   #             print('conflicting cols {} {} row {}'.format(col,col2,index_of_highest_in_col))
                return(col,col2,index_of_highest_in_col) #col, col2 have same val
    return None #no two cols of first row have same val


def test_compare_bb_dicts():
    img = '/home/jeremy/projects/core/images/2017-07-06_09-15-41-308.jpeg'
    gt = {   "data" : [
    { "object" : "Van",
      "bbox" : [1428,466, 98, 113 ]     },
    { "object" : "vw",
      "bbox" : [1306, 485, 83,64 ]     },
    { "object" : "vw",
      "bbox" : [1095,453,103,68 ]     },
    { "object" : "vw",
      "bbox" : [1204, 479, 96, 59 ]     },
    { "object" : "vw",
      "bbox" : [1010, 468, 79, 42 ]     },
    { "object" : "Van",
      "bbox" : [760, 864,586,158 ]      },
    { "object" : "sign",
      "bbox" : [750,440,270,180 ]     }  ] }

    guess =  {   "data" : [
    { "object" : "Van",
      "bbox" : [1400,500, 70, 70 ],'confidence':0.91     },
    { "object" : "Van",
      "bbox" : [1440,490, 80, 90 ],'confidence':0.86     },
    { "object" : "vw",
      "bbox" : [1300, 385, 40,50 ] ,'confidence':0.84    },
    { "object" : "Van",
      "bbox" : [1000,433,103,68 ] ,'confidence':0.92    },
    { "object" : "vw",
      "bbox" : [1200, 450, 180, 100 ] ,'confidence':0.85     },
    { "object" : "Van", 'confidence':0.7,
      "bbox" : [1306, 485, 83,64 ]     },
    { "object" : "vw",
      "bbox" : [1100, 490, 30, 50 ]  ,'confidence':0.83    },
    { "object" : "ferrari",
      "bbox" : [1060, 350, 30, 60 ] ,'confidence':0.82     },
    { "object" : "Van",
      "bbox" : [750, 869,586,158 ],'confidence':0.7  },
    { "object" : "Van",
      "bbox" : [740, 840,570,140 ],'confidence':0.81  }  ] }

    img_arr = cv2.imread(img)
  #  pdb.set_trace()
    compare_bb_dicts_class_by_class(gt,guess,img_arr = img_arr)

def get_classes_in_dict(dict,dict_format={'data':'data','object':'object'}):
    classes = []
#    print('looking at : '+str(dict))
    annotations = dict[dict_format['data']]
    for detection in annotations:
        if not detection[dict_format['object']] in classes:
            classes.append(detection[dict_format['object']])
    classes.sort()
    return classes

def get_classes_in_dicts(detection_dicts,dict_format={'data':'data','object':'object'}):
    print('***********/nCALLING GET CLASSES IN DICTS')
    classes=[]
    for dict in detection_dicts:
        print('looking at dict:'+str(dict))
        if not dict_format['data'] in dict:
            logging.warning('did not find annotations kw {} in detection {}'.format(dict_format['data'],dict))
            continue
        dict_classes = get_classes_in_dict(dict,dict_format=dict_format)
        for cl in dict_classes:
            if not cl in classes:
                classes.append(cl)
    classes.sort()
    return classes

def mAP_and_iou(gt_detections,guess_detections,dict_format={'data':'data','bbox':'bbox','object':'object','confidence':'confidence'}):

    gt_classes = get_classes_in_dicts(gt_detections,dict_format['object'])
    guess_classes = get_classes_in_dicts(guess_detections,dict_format['object'])

def get_results_and_analyze(trainfile='/mnt/hls/voc_rio_udacity_kitti_insecam_shuf_no_aug_test.txt',n_tests=1000,testdir='/data/jeremy/image_dbs/hls/voc_rio_udacity_kitti_insecam_shuf_no_aug_test/'):
    with open(trainfile,'r') as fp:
        lines = fp.readlines()
    if n_tests>len(lines):
        n_tests = len(lines)
    lines=lines[0:n_tests]
    for line in lines :
        imgfile = line.strip('\n')
        if testdir is not None:
            img_base=os.path.basename(imgfile)
            imgfile=os.path.join(testdir,img_base)
        if not os.path.exists(imgfile):
            logging.warning('image file {} not found, continuing'.format(imgfile))
            continue
        labelfile = imgfile.replace('.jpg','.txt').replace('.jpeg','.txt').replace('.png','.txt')
        if not os.path.exists(labelfile):
            logging.warning('label file {} not foind, continuing'.format(labelfile))
            continue
        results = bb_results.bb_output_yolo_using_api(imgfile,CLASSIFIER_ADDRESS=constants.YOLO_HLS_CLASSIFIER_ADDRESS,roi=None,get_or_post='GET',query='file')
        label_json = read_various_training_formats.yolo_to_tgdict(labelfile)
        print results
        print label_json



def precision_accuracy_recall(caffemodel,solverproto,outlayer='label',n_tests=100):
    #TODO dont use solver to get inferences , no need for solver for that
    caffe.set_mode_gpu()
    caffe.set_device(1)

    workdir = './'
    snapshot = 'snapshot'
#    caffemodel = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_240000.caffemodel'
#    caffemodel = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_340000.caffemodel'
    model_base = caffemodel.split('/')[-1]
    p_all = []
    r_all = []
    a_all = []
    n_all = []
#    for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.92,0.95,0.98]:
    thresh = [0.1,0.5,0.6,0.7,0.8,0.9,0.95]
#    thresh = [0.1,0.5,0.95]
    protoname = solverproto.replace('.prototxt','')
    netname = get_netname(solverproto)
    if netname:
        dir = 'multilabel_results-'+netname+'_'+model_base.replace('.caffemodel','')
        dir = dir.replace('"','')  #remove quotes
        dir = dir.replace(' ','')  #remove spaces
        dir = dir.replace('\n','')  #remove newline
        dir = dir.replace('\r','')  #remove return
    else:
        dir = 'multilabel_results-'+protoname+'_'+model_base.replace('.caffemodel','')
        dir = dir.replace('"','')  #remove quotes
        dir = dir.replace(' ','')  #remove spaces
        dir = dir.replace('\n','')  #remove newline
        dir = dir.replace('\r','')  #remove return

    print('dir to save stuff in : '+str(dir))
    Utils.ensure_dir(dir)
#    open_html(model_base,dir=dir)
    positives = True
    for t in thresh:
        p,r,a,tp,tn,fp,fn = check_accuracy(solverproto, caffemodel, threshold=t, num_batches=n_tests,outlayer=outlayer)
        p_all.append(p)
        r_all.append(r)
        a_all.append(a)
        n_occurences = [tp[i]+fn[i] for i in range(len(tp))]
        n_all.append(n_occurences)
        write_textfile(p,r,a,tp,tn,fp,fn,t,model_base,dir=dir)
        write_html(p,r,a,n_occurences,t,model_base,positives=positives,dir=dir,tp=tp,tn=tn,fp=fp,fn=fn)
        positives = False
    close_html(model_base,dir=dir)

    p_all_np = np.transpose(np.array(p_all))
    r_all_np = np.transpose(np.array(r_all))
    a_all_np = np.transpose(np.array(a_all))
    labels = constants.web_tool_categories
    plabels = [label + 'precision' for label in labels]
    rlabels = [label + 'recall' for label in labels]
    alabels = [label + 'accuracy' for label in labels]

    important_indices = [3,5,7,10,11,13,17]
    #cardigan  dress footwear jeans pants skirt top
    #['bag', 'belt', 'blazer','cardigan','coat','dress', 'eyewear', 'footwear', 'hat','jacket',
     #                  'jeans','pants','shorts', 'skirt','stocking','suit','sweater','top','scarf','womens_swimwear_bikini',
      #                 'womens_swimwear_nonbikini']

    p_important = [p_all_np[i] for i in important_indices]
    r_important = [r_all_np[i] for i in important_indices]
    a_important = [a_all_np[i] for i in important_indices]
    labels_important = [labels[i] for i in important_indices]
    for i in range(len(important_indices)):
        print(constants.web_tool_categories[i]+' p:'+str(p_important[i])+' r:'+str(r_important[i])+' a:'+str(a_important[i]) )
    thresh_all_np = np.array(thresh)
    print('shape:'+str(p_all_np.shape))
    print('len:'+str(len(p_important)))

    markers = [ '^','<','v','^','8','o',   '.','x','|',
                          '+', 0, '4', 3,4, 'H', '3', 'p', 'h', '*', 7,'', 5, ',', '2', 1, 6, 's', 'd', '1','_',  2,' ', 'D']
    markers = ['.','x','|', '^',
                '+','<',
                0,'v',
               '4', 3,'^',
                '8',
                4,'o',
                'H', '3', 'p',  '*','h',
               7,'', 5, ',', '2', 1, 6, 's', 'd', '1','_',  2,' ', 'D']
    markers_important = ['^','<','v','^', '8','o','H', '3', 'p',  '*','h']


    for i in range(len(p_important)):
        plt.subplot(311)
        print('plotting {} vs {}'.format(p_all_np[i,:],thresh_all_np))
        plt.plot(thresh_all_np,p_important[i],label=labels_important[i],linestyle='None',marker=markers_important[i])
        plt.subplot(312)   #
        plt.plot(thresh_all_np,r_important[i],label=labels_important[i],linestyle='None',marker=markers_important[i])
        plt.subplot(313)
        plt.plot(thresh_all_np,a_important[i],label=labels_important[i],linestyle='None',marker=markers_important[i])
#        plt.plot(thresh_all_np,a_all_np[i],label=labels_important[i],linestyle='None',marker=markers_important[i])
#        plt.plot(thresh_all_np,p_all_np[i,:],label=labels[i],marker=markers[i])
#        plt.subplot(312)   #
#        plt.plot(thresh_all_np,r_all_np[i,:],label=labels[i],linestyle='None',marker=markers[i])
#        plt.subplot(313)
#        plt.plot(thresh_all_np,a_all_np[i,:],label=labels[i],linestyle='None',marker=markers[i])
    plt.subplot(311)
    plt.title('results '+model_base)
    plt.xlabel('threshold')
    plt.ylabel('precision')
    plt.grid(True)
    plt.ylim((0,1))
    plt.subplot(312)   #
    plt.xlabel('threshold')
    plt.ylabel('recall')
    plt.grid(True)
    plt.ylim((0,1))
    plt.subplot(313)
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.ylim((0,1))
    plt.grid(True)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.1))
    plt.show()#

    figname = os.path.join(dir,model_base+'.png')
    print('saving figure:'+str(figname))
    plt.savefig(figname, bbox_inches='tight')
#
    summary_html(dir)
  #  print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 10,batch_size = 20))

if __name__ == " __main__":
    print('main')
    test_compare_bb_dicts()