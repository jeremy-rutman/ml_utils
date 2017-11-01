__author__ = 'jeremy' #ripped from tutorial at http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/pascal-multilabel-with-datalayer.ipynb

import sys
import os
import datetime
import numpy as np
import os.path as osp
import urllib
from copy import copy
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
import cv2
import argparse

from trendi import constants
from trendi.utils import imutils
from trendi import Utils
from trendi.classifier_stuff.caffe_nns.caffe_utils import get_netname

from trendi.paperdoll import neurodoll_falcon_client as nfc
from trendi.classifier_stuff.caffe_nns import jrinfer


# matplotlib inline
def setup():
    lt.rcParams['figure.figsize'] = (6, 6)

    caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
    sys.path.append(caffe_root + 'python')
    sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
    sys.path.append("pycaffe") # the tools file is in this folder

    # set data root directory, e.g:
    pascal_root = osp.join(caffe_root, 'data/pascal/VOC2012')

    # these are the PASCAL classes, we'll need them later.
    classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

    # make sure we have the caffenet weight downloaded.
    if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print("Downloading pre-trained CaffeNet model...")
    #    !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet

# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# main netspec wrapper
def caffenet_multilabel(data_layer_params, datalayer):
    # setup the python data layer
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'pascal_multilabel_datalayers', layer = datalayer,
                               ntop = 2, param_str=str(data_layer_params))

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output=20)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)

    return str(n.to_proto())

def makenet():
    workdir = './pascal_multilabel_with_datalayer'
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
    solverprototxt.sp['display'] = "1"
    solverprototxt.sp['base_lr'] = "0.0001"
    solverprototxt.write(osp.join(workdir, 'solver.prototxt'))

    # write train net.
    with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
        # provide parameters to the data layer as a python dictionary. Easy as pie!
        data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'train', pascal_root = pascal_root)
        f.write(caffenet_multilabel(data_layer_params, 'PascalMultilabelDataLayerSync'))

    # write validation net.
    with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
        data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'val', pascal_root = pascal_root)
        f.write(caffenet_multilabel(data_layer_params, 'PascalMultilabelDataLayerSync'))

    solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
    solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    solver.test_nets[0].share_with(solver.net)
    solver.step(1)


    ## check images loaded by batchloader
    transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
    image_index = 0 # First image in the batch.
    plt.figure()
    plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
    gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)
    plt.title('GT: {}'.format(classes[np.where(gtlist)]))
    plt.axis('off');

def hamming_distance(gt, est):
    #this is actually hamming similarity not distance
#    print('calculating hamming for \ngt :'+str(gt)+'\nest:'+str(est))
    if est.shape[0] ==1:
        print('have to do reshape')
        print('gt shape before {} est shape before {}'.format(gt.shape,est.shape))
        l=est.shape[1]
        est = est.reshape(l)
        print('gt shape after {} est shape after {}'.format(gt.shape,est.shape))
        print('gt {} est {}'.format(gt,est))
    if est.shape != gt.shape:
        print('shapes dont match')
        return 0
    else:
        pass
#        print('shapes DO match')

    hamming_similarity = sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))
 #   print('hamming = '+str(hamming_similarity))
    return hamming_similarity

def update_confmat_combine_cats(gt,est,tp,tn,fp,fn):
#    print('gt {} \nest {} sizes tp {} tn {} fp {} fn {} '.format(gt,est,tp.shape,tn.shape,fp.shape,fn.shape))
    pantsindex = constants.web_tool_categories.index('pants')
    jeansindex = constants.web_tool_categories.index('jeans')
    for i in range(len(gt)):
        #combine jeans and pants, consider also doing cardigan an sweater
        if i == pantsindex or i == jeansindex:
            if gt[pantsindex] == 1 or gt[jeansindex]==1:
                if est[pantsindex]==1 or est[jeansindex] == 1: # true positive
                    tp[i] += 1
                else:   # false negative
                    fn[i] += 1
            else:
                if est[pantsindex] or est[jeansindex]: # false positive
                    fp[i] += 1
                else:   # true negative
                    tn[i] += 1
        else:
            if gt[i] == 1:
                if est[i]: # true positive
                    tp[i] += 1
                else:   # false negative
                    fn[i] += 1
            else:
                if est[i]: # false positive
                    fp[i] += 1
                else:   # true negative
                    tn[i] += 1
#        print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    return tp,tn,fp,fn

def update_confmat(gt,est,tp,tn,fp,fn,thresh=0.5):
#    print('gt {} \nest {} sizes tp {} tn {} fp {} fn {} '.format(gt,est,tp.shape,tn.shape,fp.shape,fn.shape))
    for i in range(len(gt)):
        #combine jeans and pants, consider also doing cardigan an sweater
        if gt[i] == 1:
            if est[i]>thresh: # true positive
                tp[i] += 1
            else:   # false negative
                fn[i] += 1
        else:
            if est[i]>thresh: # false positive
                fp[i] += 1
            else:   # true negative
                tn[i] += 1
#        print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    return tp,tn,fp,fn

def precision_recall_accuracy(confmat,class_to_analyze):
    npconfmat = np.array(confmat)
    tp = npconfmat[class_to_analyze,class_to_analyze]
    fn = np.sum(npconfmat[class_to_analyze,:]) - tp
    fp = np.sum(npconfmat[:,class_to_analyze]) - tp
    tn = np.sum(npconfmat[:,:]) - tp -fn - fp
    print('confmat:'+str(confmat))
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    accuracy = float(tp+tn)/(tp+fp+tn+fn)
    print('prec {} recall {} acc {}'.format(precision,recall,accuracy))
    return precision, recall, accuracy


def test_confmat():
    gt=[True,False,1,0]
    ests=[[True,False,0,0],
          [0,0,1,0],
          [1,0,0,1],
        [ True,0,True,0]]
    tp = [0,0,0,0]
    tn = [0,0,0,0]
    fp = [0,0,0,0]
    fn = [0,0,0,0]
    tp_sum = tn_sum = fp_sum = fn_sum = [0,0,0,0]
    for e in ests:
        #update_confmat(gt,e,tp,tn,fp,fn)
        tp,tn,fp,fn = update_confmat(gt,e,tp,tn,fp,fn)
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    gt=[0,1,1,0]
    ests=[[0,1,0,1],
          [0,1,1,1],
          [1,0,0,1],
          [1,0,1,0]]
    tp_sum = tn_sum = fp_sum = fn_sum = [0,0,0,0]
    for e in ests:
        #update_confmat(gt,e,tp,tn,fp,fn)
        tp,tn,fp,fn = update_confmat(gt,e,tp,tn,fp,fn)
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))

def check_acc(net, num_samples, batch_size = 1,threshold = 0.5,gt_layer='labels',estimate_layer='prob',labels=constants.web_tool_categories_v2):
    #this is not working foir batchsize!=1, maybe needs to be defined in net
    blobs = [ k for k in net.blobs.keys()]
    print('all blobs:'+str(blobs))
    params = [ k for k in net.params.keys()]
    print('all params:'+str(params))
    acc = 0.0 #
    baseline_acc = 0.0
    n = 0

    first_time = True
    for t in range(num_samples):
        net.forward()
        gts = net.blobs[gt_layer].data
#        ests = net.blobs['score'].data > 0  ##why 0????  this was previously not after a sigmoid apparently
        ests = net.blobs[estimate_layer].data > threshold
        ests = np.array([y*1 for y in ests])
        print('net estimate_layer output:'+str(net.blobs[estimate_layer].data))
        print('net score output:'+str(net.blobs[gt_layer].data))
  #      print('xxx gts shape {} ests shape {} '.format(gts.shape,ests.shape))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            baseline_est = np.zeros_like(est)
  #          print('yyy gts shape {} ests shape {} bl shape {}:'.format(gts.shape,ests.shape,baseline_est.shape))
            if est.shape != gt.shape:
                print('shape mismatch')
                continue
            if first_time == True:
                first_time = False
                tp = np.zeros_like(gt)
                tn = np.zeros_like(gt)
                fp = np.zeros_like(gt)
                fn = np.zeros_like(gt)
            tp,tn,fp,fn = update_confmat(gt,est,tp,tn,fp,fn)
            print('tp {}\ntn {}\nfp {}\nfn {}'.format(tp,tn,fp,fn))
            print('gt:'+str([int(x) for x in gt]))  #turn to int since print as float takes 2 lines
            print('est:'+str(est))
            h = hamming_distance(gt, est)
            baseline_h = hamming_distance(gt,baseline_est)
#            print('gt {} est {} (1-hamming) {}'.format(gt,est,h))
            sum = np.sum(gt)
            acc += h
            baseline_acc += baseline_h
            n += 1
    print('len(gts) {} len(ests) {} batchsize {} acc {} baseline {}'.format(len(gts),len(ests),batch_size,acc/n,baseline_acc/n))
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    full_rec = [float(tp[i])/(tp[i]+fn[i]) for i in range(len(tp))]
    full_prec = [float(tp[i])/(tp[i]+fp[i]) for i in range(len(tp))]
    full_acc = [float(tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i]) for i in range(len(tp))]
    print('THRESHOLD '+str(threshold))
    print('precision {}\nrecall {}\nacc {}\navgacc {}'.format(full_prec,full_rec,full_acc,acc/n))
    return full_prec,full_rec,full_acc,tp,tn,fp,fn

def check_acc_nonet(ground_truths,estimates,threshold=0.5):
    acc = 0.0 #
    baseline_acc = 0.0
    n = 0
    first_time = True
    for gt, est in zip(ground_truths, estimates): #for each ground truth and estimated label vector
        if est.shape != gt.shape:
            print('shape mismatch')
            return

        if first_time == True:
            first_time = False
            tp = np.zeros_like(gt)
            tn = np.zeros_like(gt)
            fp = np.zeros_like(gt)
            fn = np.zeros_like(gt)
            baseline_est = np.zeros_like(est)
        #binarize the estimate which can come in as floating
    #    print('type of est:'+str(type(est))+' shape:'+str(est.shape))
        if type(est) is np.uint8 or type(est) is np.int or type(est) is np.float or type(est) is np.float64 or type(est) is np.ndarray:
            print('g ot np arr')
            est = est>threshold
        else:
            est = np.array([e>threshold for e in est])
        tp,tn,fp,fn = update_confmat(gt,est,tp,tn,fp,fn)
        print('tp {}\ntn {}\nfp {}\nfn {}'.format(tp,tn,fp,fn))
        print('gt:'+str([int(x) for x in gt]))  #turn to int since print as float takes 2 lines
        print('est:'+str(est))
        h = hamming_distance(gt, est)
        baseline_h = hamming_distance(gt,baseline_est)
        #            print('gt {} est {} (1-hamming) {}'.format(gt,est,h))
        sum = np.sum(gt)
        acc += h
        baseline_acc += baseline_h
        n += 1

    print('len(gts) {} len(ests) {} acc {} baseline {}'.format(len(ground_truths),len(estimates),acc/n,baseline_acc/n))
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    full_rec = [float(tp[i])/(tp[i]+fn[i]) for i in range(len(tp))]
    full_prec = [float(tp[i])/(tp[i]+fp[i]) for i in range(len(tp))]
    full_acc = [float(tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i]) for i in range(len(tp))]
    print('THRESHOLD '+str(threshold))
    print('precision {}\nrecall {}\nacc {}\navgacc {}'.format(full_prec,full_rec,full_acc,acc/n))
    print('tp {}\ntn {}\nfp {}\nfn {}'.format(tp,tn,fp,fn))

    return full_prec,full_rec,full_acc,tp,tn,fp,fn

#train
def train():
    for itt in range(6):
        solver.step(100)
        print 'itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 50))

def check_baseline_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = np.zeros((batch_size, len(gts)))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            print('gt {} est {} '.format(gt,est))
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


def results_unk():#prediction results
    test_net = solver.test_nets[0]
    for image_index in range(5):
        plt.figure()
        plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
        gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
        estlist = test_net.blobs['score'].data[image_index, ...] > 0
        plt.title('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[np.where(estlist)]))
        plt.axis('off')

def check_accuracy(proto,caffemodel,num_batches=200,batch_size=1,threshold = 0.5,outlayer='label'):
    print('checking accuracy of net {} using proto {}'.format(caffemodel,proto))
#    solver = caffe.SGDSolver(solverproto)
     # Make classifier.
    #classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
    #                          image_dims=image_dims, mean=mean,
    #                         input_scale=input_scale, raw_scale=raw_scale,
    #                          channel_swap=channel_swap)
    net = caffe.Net(proto,caffemodel, caffe.TEST)
    caffe.set_mode_gpu()
    caffe.set_device(1)

#    solver.net.copy_from(caffemodel)
#    solver.test_nets[0].share_with(solver.net)
#    solver.step(1)
#    precision,recall,accuracy,tp,tn,fp,fn = check_acc(solver.test_nets[0], num_batches=num_batches,batch_size = batch_size, threshold=threshold)
    precision,recall,accuracy,tp,tn,fp,fn = check_acc(net, num_batches=num_batches,batch_size = batch_size, threshold=threshold)
    return precision,recall,accuracy,tp,tn,fp,fn

def check_accuracy_hydra_using_multilabel(proto,caffemodel,num_images=5,
                         multilabel_file='/data/jeremy/image_dbs/tamara_berg_street_to_shop/tb_cats_from_webtool_round2_train.txt',
                         outlayers=['fc4_0','fc4_1'],cats=['skirt','pants'],save_imgs=True,show_imgs=True):
    with open(multilabel_file,'r') as fp:
        lines = fp.readlines()
        fp.close()
    files = [l.split()[0] for l in lines]
    labels = [[int(i) for i in l.split()[1:]] for l in lines]
    if num_images is not None:
        files = files[0:num_images]
        labels = labels[0:num_images]
    results = jrinfer.infer_many_hydra(files,proto,caffemodel,out_dir='./',dims=(224,224),output_layers=outlayers)
    print('all reslts from jrinfer.infer_many_hydra:'+str(results))
    indices = [constants.web_tool_categories_v2.index(cat) for cat in cats]
    print('cats {} indices {}'.format(cats,indices))
    reduced_labels = []
    for l in labels:  #tkae only selected elements from multilabel result
        print('l b4 '+str(l))
        reduced_l = np.zeros(len(outlayers))
        for i in range(len(outlayers)):
            reduced_l[i]=l[indices[i]]
        reduced_labels.append(reduced_l)
        print('l after '+str(reduced_l))
    reduced_labels = np.array(reduced_labels,dtype=np.uint8)
    reduced_results = []
    n=0
    for r in results: #take results from [p1,p2,...] to index of winner (largest)
        print('r b4 '+str(r))
        reduced_r=np.zeros(len(outlayers),dtype=np.uint8)
        i=0
        for i in range(len(r)):
            reduced_r[i]=np.argmax(r[i])
            i=i+1
        reduced_results.append(reduced_r)
        print('r after '+str(reduced_r))
        if save_imgs:
            img_arr = cv2.imread(files[n])
            ml_str=''
            for j in range(len(reduced_labels[n])):
                ml_str = ml_str+cats[j]+'gt'+str(reduced_labels[n][j])+'est'+str(reduced_results[n][j])+'_'
            name = 'img'+str(n)+ml_str[:-1]+'.jpg'
            print('lbl {} est {}'.format(reduced_labels[n],reduced_results[n]))
            print('saving to '+ml_str)
            cv2.imwrite(name,img_arr)
        if show_imgs:
            cv2.imshow(files[n]+'_'+str(n),cv2.imread(files[n]))
            cv2.waitKey(0)
        n=n+1

    reduced_results = np.array(reduced_results,dtype=np.uint8)
    print('labels {} \n results {}'.format(reduced_labels,reduced_results))
    precision,recall,accuracy,tp,tn,fp,fn = check_acc_nonet(reduced_labels,reduced_results,threshold=0.5)
    return precision,recall,accuracy,tp,tn,fp,fn

def check_accuracy_hydra_using_single_label(proto,caffemodel,num_images=10,
                         label_file='/data/jeremy/image_dbs/tamara_berg_street_to_shop/binary/dress_filipino_labels_balanced_test_250x250.txt',
                         outlayers=['fc4_0','fc4_1'],save_imgs=True,hydra_head_index=1,n_classes=2):
    '''
    check individual outputs using files of single-label ground truths
    :param proto:
    :param caffemodel:
    :param num_images:
    :param multilabel_file:
    :param outlayers:
    :return:
    '''
    with open(label_file,'r') as fp:
        lines = fp.readlines()
        fp.close()
    files = [l.split()[0] for l in lines]
    labels = [int(l.split()[1]) for l in lines]
    if num_images is not None:
        files = files[0:num_images]
        labels = labels[0:num_images]
    results = jrinfer.infer_many_hydra(files,proto,caffemodel,out_dir='./',dims=(224,224),output_layers=outlayers)
    print('all results from jrinfer.infer_many_hydra:'+str(results))
    reduced_labels = np.array(labels,dtype=np.uint8)
    reduced_results = []
    i=0
    for r in results: #take results from [p1,p2,...] to index of winner (largest)
        print('r b4 '+str(r))
        reduced_r=np.argmax(r[hydra_head_index])
        reduced_results.append(reduced_r)
        print('r after '+str(reduced_r))
        if save_imgs:
            img_arr = cv2.imread(files[i])
            name = 'img'+str(i)+'gt_'+str(reduced_labels[i])+'_est_'+str(reduced_r)+'_head_'+str(hydra_head_index)+'.jpg'
            cv2.imwrite(name,img_arr)
        i=i+1
    reduced_results = np.array(reduced_results,dtype=np.uint8)
    print('labels {} \n results {}'.format(reduced_labels,reduced_results))
    confmat = np.zeros([n_classes,n_classes])
    for gt,est in zip(reduced_labels,reduced_results):
        confmat[gt][est]+=1
    precision_recall_accuracy(confmat,1)  #pra(confmat,class_to_analyze)

def multilabel_infer_one(url):
    image_mean = np.array([104.0,117.0,123.0])
    input_scale = None
    channel_swap = [2, 1, 0]
    raw_scale = 255.0
    print('loading caffemodel for neurodoll (single class layers)')

    start_time = time.time()
    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)   #
    in_ = np.array(in_, dtype=np.float32)   #.astype(float)
    if len(in_.shape) != 3:  #h x w x channels, will be 2 if only h x w
        print('got 1-chan image, turning into 3 channel')
        #DEBUG THIS , ORDER MAY BE WRONG [what order? what was i thinking???]
        in_ = np.array([in_,in_,in_])
    elif in_.shape[2] != 3:  #for rgb/bgr, some imgages have 4 chan for alpha i guess
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR
#    cv2.imshow('test',in_)
    in_ -= np.array((104,116,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = net.blobs['siggy'].data[0][category_index] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out shape {}'.format(min,max,out.shape))
    out = out*255
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out after scaling  {}'.format(min,max,out.shape))
    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
#    outname = os.path.basename(imagename)
#    outname = outname.split('.jpg')[0]+'.bmp'
#    outname = os.path.join(out_dir,outname)
#    print('outname:'+outname)
#    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('infer_one elapsed time:'+str(elapsed_time))
 #   cv2.imshow('out',out.astype(np.uint8))
 #   cv2.waitKey(0)
    return out.astype(np.uint8)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format

    if url.count('jpg') > 1:
        return None

    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return new_image

def get_multilabel_output(url_or_np_array,required_image_size=(227,227),output_layer_name='prob'):


    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)
    in_ = np.array(in_, dtype=np.float32)   #.astype(float)
    if len(in_.shape) != 3:  #h x w x channels, will be 2 if only h x w
        print('got 1-chan image, turning into 3 channel')
        #DEBUG THIS , ORDER MAY BE WRONG [what order? what was i thinking???]
        in_ = np.array([in_,in_,in_])
    elif in_.shape[2] != 3:  #for rgb/bgr, some imgages have 4 chan for alpha i guess
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR : since this is loaded nby cv2 its unecessary
#    cv2.imshow('test',in_)
    in_ -= np.array((104,116,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    multilabel_net.blobs['data'].reshape(1, *in_.shape)
    multilabel_net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    multilabel_net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = multilabel_net.blobs[output_layer_name].data[0] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('out  {}'.format(out))

def get_multilabel_output_using_nfc(url_or_np_array):
    print('starting get_multilabel_output_using_nfc')
    multilabel_dict = nfc.pd(url_or_np_array, get_multilabel_results=True)
    logging.debug('get_multi_output:dict from falcon dict:'+str(multilabel_dict))
    if not multilabel_dict['success']:
        logging.warning('did not get nfc pd result succesfully')
        return
    multilabel_output = multilabel_dict['multilabel_output']
    logging.debug('multilabel output:'+str(multilabel_output))
    return multilabel_output #

def test_multilabel_output_on_testset(testfile,outdir='./'):
    '''
    this takes an images+multilabels file and determines accuracy
    :param testfile: images+labels n testset
    :return:acc/precision/recall
    '''
    #get list of images, labels from file
    img_files=[]
    label_vecs=[]
    with open(testfile,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            imgfilename = line.split()[0]
            try:
                vals = line.split()[1:]
                n_labels = len(vals)
                label_vec = [int(i) for i in vals]
            except:
                logging.debug('got something that coulndt be turned into a string in the following line from file '+testfile)
                logging.debug(line)
                logging.debug('error:'+str(sys.exc_info()[0])+' , skipping line')
                continue
            label_vec = np.array(label_vec)
            n_labels = len(label_vec)
            if n_labels == 1:
        #                  print('length 1 label')
                label_vec = label_vec[0]
            img_files.append(imgfilename)
            label_vecs.append(label_vec)

    estimates=[]
    for imgfile in img_files:
        img_arr = cv2.imread(imgfile)
        if img_arr is None:
            logging.info('could not read '+str(imgfile))
            estimates.append(None)
            continue
        ml_output=get_multilabel_output_using_nfc(img_arr)
        print('ml output:'+str(ml_output))
        mlfilename = os.path.basename(imgfile).replace('.txt','_mloutput.txt')
        mlfilename = os.path.join(outdir,mlfilename)
        with(open(mlfilename,'a')) as fp:
            fp.write(ml_output)
        fp.close()
        estimates.append(ml_output)
    print('checking accuracy')
    check_acc_nonet(label_vecs,estimates)

def multilabel_output_on_testfile(testfile=None,testdir=None,filter='.jpg',outdir='./',estimates_file='estimates.txt'):
    if testfile is not None:
        img_files = []
        with open(testfile,'r') as fp:
            lines = fp.readlines()
            for line in lines:
                imgfilename = line.split()[0]
                img_files.append(imgfilename)
    elif testdir is not None:
        img_files = [os.path.join(testdir,f) for f in os.listdir(testdir) if filter in f]
    n = len(img_files)
    estimates=[]
    i=0
    for imgfile in img_files:
        print('doing {} ({}/{})'.format(imgfile,i,n))
        img_arr = cv2.imread(imgfile)
        if img_arr is None:
            logging.info('could not read '+str(imgfile))
            estimates.append(None)
            continue
        ml_output=get_multilabel_output_using_nfc(img_arr)
        print('ml output:'+str(ml_output))
        mlfilename = os.path.basename(imgfile).replace('.jpg','_mloutput.txt')
        mlfilename = os.path.join(outdir,mlfilename)
        logging.info('mlfilename:'+str(mlfilename))
        with(open(mlfilename,'a')) as fp:
            for e in ml_output:
                fp.write(str(round(e,3))+' ')
            fp.write('\n')
            fp.close()
        estimates.append(ml_output)
        i=i+1

    with open(estimates_file,'a') as fp:
        for imgfile,estimate in zip(img_files,estimates):
            fp.write(imgfile+' ')
            for e in estimate:
                fp.write(str(round(e,3))+' ')
            fp.write('\n')#

def open_html(modelname,dir=None,solverproto='',caffemodel='',classlabels = constants.web_tool_categories,name=None):
    model_base = os.path.basename(modelname)
    if dir is not None:
        Utils.ensure_dir(dir)
        htmlname = os.path.join(dir,model_base+'results.html')
    else:
        htmlname = os.path.join(model_base,'results.html')
    if name is not None:
        htmlname = name
    with open(htmlname,'a') as g:
        g.write('<!DOCTYPE html>')
        g.write('<html>')
        g.write('<head>')
        g.write('<title>')
        dt=datetime.datetime.today()
        g.write(model_base+' '+dt.isoformat())
        g.write('</title>')
        g.write('solver:'+solverproto+'\n'+'<br>')
        g.write('model:'+caffemodel+'\n'+'<br>')
        g.write('</head>')
#        g.write('categories: '+str(constants.web_tool_categories)+'<br>'+'\n')
        g.write('<table style=\"width:100%\">\n')
        g.write('<tr>\n')
        g.write('<th>')
        g.write('metric')
        g.write('</th>\n')
        g.write('<th>')
        g.write('fw avg.')
        g.write('</th>\n')
        if classlabels:
            for i in range(len(classlabels)):
                g.write('<th>')
                g.write(classlabels[i])
                g.write('</th>\n')
        g.write('</tr>\n')

#        g.write('</table><br>')

def close_html(model_base,dir=None):
    if dir is None:
        protoname = solverproto.replace('.prototxt','')
        dir = 'multilabel_results-'+protoname+'_'+model_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    htmlname = os.path.join(dir,model_base+'results.html')
    with open(htmlname,'a') as g:
        g.write('</table><br>')
        plotfilename = 'multilabel_results'+model_base+'.png'

        g.write('<a href=\"'+plotfilename+'\">plot<img src = \"'+plotfilename+'\" style=\"width:300px\"></a>')
        g.write('</html>')

def summary_html(dir):
    htmlfiles = [f for f in os.listdir(dir) if '.html' in f]
    htmlname = os.path.join(dir,'summary.html')
    with open(htmlname,'w') as g:
        g.write('<!DOCTYPE html><br>\n')
        g.write('<html><br>\n')
        g.write('<head><br>\n')
        g.write('<title><br>\n')
        g.write('multilabel accuracy/precision/recall results')
        g.write('</title><br>\n')
        g.write('</head>\n')
        g.write('<br>\n')
        for h in htmlfiles:
            g.write('<a href=\"'+ h+'\"> '+h+'</a><br>\n')
        g.write('</html>')

#        g.write('categories: '+str(constants.web_tool_categories)+'<br>'+'\n')


def write_html(p,r,a,n,threshold,modelname,positives=False,dir=None,name=None,classlabels=None):
    model_base = os.path.basename(modelname)

    if dir is not None:
        Utils.ensure_dir(dir)
        htmlname = os.path.join(dir,model_base+'results.html')
    else:
        htmlname = os.path.join(model_base,'results.html')
    if name is not None:
        htmlname = name

    open_html(model_base,dir=dir,classlabels=classlabels)

    with open(htmlname,'a') as g:
        fwavp = 0
        fwavr = 0
        fwava = 0
        n_p=0
        n_r=0
        n_a=0
        fwavn = 0
        n_sum = 0
        #calculate frequency-weighted averages
        for i in range(len(p)):
            if not np.isnan(p[i]):
                fwavp = fwavp + p[i]*n[i]
                n_p=n_p+n[i]
            if not np.isnan(r[i]):
                fwavr = fwavr + r[i]*n[i]
                n_r=n_r+n[i]
            if not np.isnan(a[i]):
                fwava = fwava + a[i]*n[i]
                n_a=n_a+n[i]
            n_sum=n_sum+n[i]
        print('n sum {} fwavp {} fwavr {} fwava {} before division np {} nr {} na {} '.format(n_sum,fwavp,fwavr,fwava,n_p,n_r,n_a))
        fwavp = fwavp/float(n_p)
        fwavr = fwavr/float(n_r)
        fwava = fwava/float(n_a)
        fwavn = n_sum/float(len(p))

        print('frequency weighted averages p {} r {} acc {} n {}'.format(fwavp,fwavr,fwava,fwavn))
        g.write('frequency weighted averages p {} r {} acc {} n {}'.format(round(fwavp,3),round(fwavr,3),round(fwava,3),round(fwavn,3)))
    #write line with n_positives
        if(positives):
            g.write('<tr>\n')
            g.write('<td>')
            g.write('n_positives')
            g.write('</td>\n')
            g.write('<td>')
            g.write(str(round(fwavn,3)))
            g.write('</td>\n')
            for i in range(len(p)):
                g.write('<td>')
                g.write(str(int(n[i])))
                g.write('</td>\n')
            g.write('</tr>\n<br>\n')

    #write line with threshold
 #       g.write('<table style=\"width:100%\">\n')
        g.write('<b>')
        g.write('<tr>\n')
        g.write('<td>')
        g.write('threshold\n')
        g.write('</td>')
        g.write('<td>')
        g.write('')
        g.write('</td>\n')
        for i in range(len(p)):
            g.write('<td>')
            g.write(str(round(threshold,3)))
            g.write('</td>\n')
        g.write('</tr>\n')
        g.write('</b>')

    #write row with precision
        g.write('<tr>\n')
        g.write('<td>')
        g.write('precision')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(fwavp,3)))
        g.write('</td>\n')
        for i in range(len(p)):
            g.write('<td>')
            g.write(str(round(p[i],3)))
            g.write('</td>\n')
        g.write('</tr>\n')

    #write row with recall
        g.write('<tr>\n')
        g.write('<td>')
        g.write('recall')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(fwavr,3)))
        g.write('</td>\n')
        for i in range(len(p)):
            g.write('<td>')
            g.write(str(round(r[i],3)))
            g.write('</td>\n')
        g.write('</tr>\n')

    #write row with accuracy
        g.write('<tr>\n')
        g.write('<td>')
        g.write('accuracy')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(fwava,3)))
        g.write('</td>\n')
        for i in range(len(p)):
            g.write('<td>')
            g.write(str(round(a[i],3)))
            g.write('</td>\n')
        g.write('</tr>\n<br>\n')

        g.write('<tr><td><br/></td></tr>')  #blank row


#        g.write('threshold = '+str(t)+'\n')

def write_textfile(p,r,a,tp,tn,fp,fn,threshold,model_base,dir=None):
    if dir is None:
        protoname = solverproto.replace('.prototxt','')
        dir = 'multilabel_results-'+protoname+'_'+model_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    fname = os.path.join(dir,model_base+'results.txt')
    with open(fname,'a') as f:
        f.write(model_base+' threshold = '+str(threshold)+'\n')
        f.write('solver:'+solverproto+'\n')
        f.write('model:'+caffemodel+'\n')
        f.write('categories: '+str(constants.web_tool_categories)+ '\n')
        f.write('precision\n')
        f.write(str(p)+'\n')
        f.write('recall\n')
        f.write(str(r)+'\n')
        f.write('accuracy\n')
        f.write(str(a)+'\n')
        f.write('true positives\n')
        f.write(str(tp)+'\n')
        f.write('true negatives\n')
        f.write(str(tn)+'\n')
        f.write('false positives\n')
        f.write(str(fp)+'\n')
        f.write('false negatives\n')
        f.write(str(fn)+'\n')
        f.close()


def get_traintest_from_proto(proto):
    print('looking for netname in '+proto)
    with open(proto,'r') as fp:
        train = None
        test = None
        traintest = None
        for line in fp:
            line = line.replace(' ','')  #line with spaces removed
            print('looking at line:'+line)
            if 'train_net:' in line and line[0] is not '#':
                train = line.replace('train_net:','').replace('"','')
                print('train:'+train)
            if 'test_net:' in line and line[0] is not '#':
                test = line.replace('test_net:','').replace('"','')
                print('test:'+test)
            if line[0:3] == 'net:'  and line[0] is not '#':
                traintest = line.replace('net:','').replace('"','')
                print('traintest:'+traintest)
        if train and test:
            return((train,test))
        elif train:
            print('got only train not test')
            return((train))
        elif test:
            print('got only test not train')
            return((test))
        elif traintest:
            return((traintest))
        else:
            return None

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



if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='multilabel accuracy tester')
    parser.add_argument('--testproto',  help='test prototxt')
    parser.add_argument('--caffemodel', help='caffmodel')
    parser.add_argument('--gpu', help='gpu #',default=0)
    parser.add_argument('--output_layer_name', help='output layer name',default='prob')
    parser.add_argument('--n_tests', help='number of examples to test',default=1000)

    caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet50_sgd_iter_120000.caffemodel'
    solverproto = '/home/jeremy/caffenets/production/ResNet-50-test.prototxt'
#    caffemodel =  '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_340000.caffemodel'
#    deployproto = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/deploy.prototxt'
    solverproto = '/home/jeremy/caffenets/multilabel/deep-residual-networks/prototxt/ResNet-101-test.prototxt'
    caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet101_sgd_iter_120000.caffemodel'
#    multilabel_net = caffe.Net(solverproto,caffemodel, caffe.TEST)

    args = parser.parse_args()
    print(args)
    if args.testproto is not None:
        solverproto = args.testproto
    if args.caffemodel is not None:
        caffemodel = args.caffemodel
#    if args.gpu is not None:
    gpu = int(args.gpu)
#    if args.output_layer_name is not None:
    outlayer = args.output_layer_name
    n_tests = int(args.n_tests)
 #   else:
 #       outlayer = 'label'
    caffe.set_mode_gpu()
    caffe.set_device(gpu)


    precision_accuracy_recall(caffemodel,solverproto,outlayer=outlayer,n_tests=n_tests)

#    t = 0.5
#    p,r,a,tp,tn,fp,fn = check_accuracy(solverproto, caffemodel, threshold=t, num_batches=n_tests,outlayer=outlayer)



