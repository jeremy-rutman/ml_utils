# coding: utf-8
__author__ = 'jeremy'

# Run the script with anaconda-python
# $ /home/<path to anaconda directory>/anaconda/bin/python LmdbClassification.py
import sys
import numpy as np
import lmdb
import caffe
from collections import defaultdict
import socket
from trendi.utils import imutils
import cv2
import matplotlib as plt
import argparse
import time
import logging
logging.basicConfig(level=logging.DEBUG)

from trendi import Utils
from trendi.classifier_stuff.caffe_nns import lmdb_utils

def confusion_vector_for_binary_classifiers(deploy_prototxt,caffe_model_file_path,test_lmdb_path_list=None,meanB=128,meanG=128,meanR=128):
    if host != 'jr-ThinkPad-X1-Carbon':
        caffe.set_mode_gpu()
    if test_lmdb_path_list is None:
        lmdb_utils.generate_binary_dbs(dir_of_dirs,filter='test')

    # Modify the paths given below
    # Extract mean from the mean image file
#    mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
  #  f = open(mean_file_binaryproto, 'rb')
   # mean_blobproto_new.ParseFromString(f.read())
   # mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    #f.close()

    # CNN reconstruction and loading the trained weights
    net = caffe.Net(deploy_prototxt_file_path, caffe_model_file_path, caffe.TEST)

    count = 0
    correct = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    lmdb_env = lmdb.open(test_lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
#        out = net.forward_all(data=np.asarray([image]) - mean_image)
#        image[:,:,0] = image[:,:,0]- meanB
  #      image[:,:,1] = image[:,:,1]- meanB
    #    image[:,:,2] = image[:,:,2] - meanB
        thedata = np.asarray([image])
        out = net.forward_all(thedata)
        plabel = int(out['prob'][0].argmax(axis=0))
        count += 1
        iscorrect = label == plabel
        correct += (1 if iscorrect else 0)
        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])

        if not iscorrect:
            print("\rError: key = %s, expected %i but predicted %i" % (key, label, plabel))
            sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
            sys.stdout.flush()

    print("\n" + str(correct) + " out of " + str(count) + " were classified correctly")
    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])

def confusion_matrix(deploy_prototxt_file_path,caffe_model_file_path,test_lmdb_path,meanB=128,meanG=128,meanR=128):
#    caffe.set_mode_gpu()

    # Modify the paths given below

    # Extract mean from the mean image file
#    mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
  #  f = open(mean_file_binaryproto, 'rb')
   # mean_blobproto_new.ParseFromString(f.read())
   # mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    #f.close()

    # CNN reconstruction and loading the trained weights
    net = caffe.Net(deploy_prototxt_file_path, caffe_model_file_path, caffe.TEST)

    count = 0
    correct = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    lmdb_env = lmdb.open(test_lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
#        out = net.forward_all(data=np.asarray([image]) - mean_image)
#        image[:,:,0] = image[:,:,0]- meanB
  #      image[:,:,1] = image[:,:,1]- meanB
    #    image[:,:,2] = image[:,:,2] - meanB
        thedata = np.asarray([image])
        out = net.forward_all(thedata)
        plabel = int(out['prob'][0].argmax(axis=0))
        count += 1
        iscorrect = label == plabel
        correct += (1 if iscorrect else 0)
        matrix[(label, plabel)] += 1
        labels_set.update([label, plabel])

        if not iscorrect:
            print("\rError: key = %s, expected %i but predicted %i" % (key, label, plabel))
            sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
            sys.stdout.flush()

    print("\n" + str(correct) + " out of " + str(count) + " were classified correctly")
    print ""
    print "Confusion matrix:"
    print "(r , p) | count"
    for l in labels_set:
        for pl in labels_set:
            print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])

def get_nn_answer(prototxt,caffemodel,mean_B=128,mean_G=128,mean_R=128,image_filename='../../images/female1.jpg',image_width=150,image_height=200):
        host = socket.gethostname()
        print('host:'+str(host))
        pc = False
        caffe.set_mode_gpu()
        if host == 'jr-ThinkPad-X1-Carbon':
            pc = True
            caffe.set_mode_cpu()
        net = caffe.Net(prototxt,caffemodel,caffe.TEST)

    #    solver = caffe.SGDSolver(proto_file_path)
        # see http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
    #    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
     #   mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        # create transformer for the input called 'data'
        trans = False
        if trans:  #do the xforms using the caffe transformer
            mu = np.array([mean_B,mean_G,mean_R])
            print 'mean-subtracted values:',  mu
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            logging.debug('transformer')
            transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
            logging.debug('transpose')
            transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
       #     transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
            logging.debug('mean')
            transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
            logging.debug('scale')
        #    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
        # set the size of the input (we can skip this if we're happy
        #  with the default; we can also change it later, e.g., for different batch sizes)
     #       net.blobs['data'].reshape(batch_size,        # batch size
     #                                 image_depth,         # 3-channel (BGR) images
     #                                image_width, image_height)  # image size is 227x227
                #possibly use cv2.imread here instead as that's how i did it in lmdb_utils
            image = caffe.io.load_image(image_filename)
            logging.debug('load')
            cv2.imshow(image_filename,image)
            logging.debug('imshow')
    #        fig = plt.figure()
    #        fig.savefig('out.png')
            transformed_image = transformer.preprocess('data', image)
            logging.debug('preprocess')
        else: #dont use transformer, rather do it myself
            img_arr = cv2.imread(image_filename)
            h,w = img_arr.shape[0:2]
            if h != image_height or w != image_width:
                img_arr = cv2.resize(img_arr,(image_height,image_width))
            if mean_B is not None and mean_G is not None and mean_R is not None:
                img_arr[:,:,0] = img_arr[:,:,0]-mean_B
                img_arr[:,:,1] = img_arr[:,:,1]-mean_G
                img_arr[:,:,2] = img_arr[:,:,2]-mean_R
            img_arr = np.divide(img_arr,255.0)
            transformed_image = img_arr.transpose((2,0,1))

    # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image
        logging.debug('netblobs')

        ### perform classification
        output = net.forward()

#        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
#        print('all '+str(output))
        n = net.blobs
        print('net '+str(n))

        output = n['output_layer'].data
        print('output '+str(output))


        #        correct += sum(solver.test_nets[0].blobs['output_layer'].data.argmax(1)
        #                       == solver.test_nets[0].blobs['label'].data)

#        print 'predicted classes:', output_prob
#        print 'predicted class is:', output_prob.argmax()


def test_net(prototxt,caffemodel, db_path):
    net = caffe.Net(prototxt, caffemodel,caffe.TEST)
    caffe.set_mode_cpu()
    lmdb_env = lmdb.open(db_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    count = 0
    correct = 0
    max_to_test = 100
    for key, value in lmdb_cursor:
        print "Count:"
        print count
        count = count + 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        print('image size'+len(image)+' shape:'+str(image.shape))
        out = net.forward_all(data=np.asarray([image]))
        predicted_labels = out['prob'][0]
        most_probable_label = out['prob'][0].argmax(axis=0)
        if label == most_probable_label[0][0]:
            correct = correct + 1
        print("Label is class " + str(label) + ", predicted class is " + str(most_probable_label[0][0]))
        if count == max_to_test:
            break
    print(str(correct) + " out of " + str(count) + " were classified correctly")

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def detect_with_scale_pyramid_and_sliding_window(image_filename_or_cv2_array,prototxt,caffemodel,mean_B=128,mean_G=128,mean_R=128,image_width=150,image_height=200,show_visual_output=False):
    caffe.set_mode_gpu()
    if host == 'jr-ThinkPad-X1-Carbon':
        caffe.set_mode_cpu()
    net = caffe.Net(prototxt,caffemodel,caffe.TEST)
    img_arr = Utils.get_cv2_img_array(image_filename_or_cv2_array)
    orig_img_arr = img_arr.copy()

    if(0):
        if mean_B is not None and mean_G is not None and mean_R is not None:
            img_arr[:,:,0] = img_arr[:,:,0]-mean_B
            img_arr[:,:,1] = img_arr[:,:,1]-mean_G
            img_arr[:,:,2] = img_arr[:,:,2]-mean_R
        img_arr = np.divide(img_arr,255.0)
        transformed_image = img_arr.transpose((2,0,1))

        net.blobs['data'].data[...] = transformed_image
        ### perform classification
        output = net.forward()

        logging.debug('orig shape '+str(img_arr.shape))
        h,w = img_arr.shape[0:2]
#    if h != image_height or w != image_width:
#        img_arr = cv2.resize(img_arr,(image_width,image_height))

# copy the image data into the memory allocated for the net


    i=0
# loop over the image pyramid
    for resized in pyramid(img_arr, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(image_width, image_height)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != image_height or window.shape[1] != image_width:
                logging.debug('got bad window shape from sliding_window')
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            img_arr2=window.copy()
            if mean_B is not None and mean_G is not None and mean_R is not None:
                img_arr2[:,:,0] = img_arr2[:,:,0]-mean_B
                img_arr2[:,:,1] = img_arr2[:,:,1]-mean_G
                img_arr2[:,:,2] = img_arr2[:,:,2]-mean_R
            img_arr2 = np.divide(img_arr2,255.0)
            transformed_image = img_arr2.transpose((2,0,1))

            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()

            n = net.blobs
            print('net '+str(n))
            output = n['output_layer'].data
            print('output '+str(output))


            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + image_width, y +image_height), (0, 255, 0), 2)
            if show_visual_output:
                cv2.imshow("sliding window", clone)
                cv2.imshow("window", window)
                cv2.waitKey(1)
                time.sleep(0.025)
            fname = 'output'+str(i)+'.jpg'
#            cv2.imwrite(fname,clone)
            i = i +1


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w,h))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def get_netname(proto):
    logging.info('looking for netname in'+str(proto))
    with open(proto,'r') as fp:
        l1 = fp.readline()
        l2 = fp.readline()
    logging.debug('line1 '+l1)
    logging.debug('line2 '+l2)
    print('l1'+l1)
    print('l2'+l2)
    if 'name' in l1:
        netname = l1[l1.find('name:')+5:] #get string after name:
        netname = netname.replace('"','')  #remove quotes
        logging.info('line1:'+l1)
        logging.info('netname:'+netname)
        return netname
    if 'name' in l2:
        netname = l2[l2.find('name:')+5:] #get string after name:
        netname = netname.replace('"','')  #remove quotes
        logging.info('line2:'+l2)
        logging.info('netname:'+netname)
        return netname
    if 'test_net' or 'train_net' in l1: #the file is prob a solverproto and refers to test/val which may have netname
        logging.info('trying to find netname in line1 '+l1)
        s = l1.split('"')
        if len(s)>1:
            fname = l1.split('"')[-2]
            return get_netname(fname)
        else:
            return None
    if 'test_net' or 'train_net' in l2:
        print('trying to find netname in line2 '+l2)
        s = l2.split('"')
        if len(s)>1:
            fname = l2.split('"')[-2]
            return get_netname(fname)
        else:
            return None
    else:
        netname = None
    return netname


host = socket.gethostname()
print('host:'+str(host))

if __name__ == "__main__":

    if host == 'jr-ThinkPad-X1-Carbon':
        pass
    else:
        prototxt = '/home/jeremy/core/classifier_stuff/caffe_nns/alexnet10_binary_dresses/my_solver.deploy.prototxt'
        caffemodel = '/home/jeremy/core/classifier_stuff/caffe_nns/alexnet10_binary_dresses/net_iter_9000.caffemodel'

        img_filename = '/home/jeremy/core/images/female1.jpg'

    parser = argparse.ArgumentParser(description='test an image yo')
    parser.add_argument('-i', '--image', help='path to image file to be analyzed yo', required=False)
    parser.add_argument('-d', '--deploy_proto', help='deploy prototxt', required=False)
    parser.add_argument('-c', '--caffemodel', help='caffe model', required=False)
    args = vars(parser.parse_args())
    print args
    args = parser.parse_args()
    print args

    if args.image:
        img_filename = args.image
    if args.deploy_proto:
        prototxt = args.deploy_proto
    if args.caffemodel:
        caffemodel = args.caffemodel

    print('img {} proto {} caffemodel {}'.format(img_filename,prototxt,caffemodel))
    #get_nn_answer(prototxt,caffemodel,mean_B=112,mean_G=123,mean_R=136,image_filename=img_filename,image_width=150,image_height=200)
    detect_with_scale_pyramid_and_sliding_window(img_filename,prototxt,caffemodel,mean_B=128,mean_G=128,mean_R=128,image_width=150,image_height=200,show_visual_output=False)

#        deploy_prototxt
#        conf_mat(deploy_prototxt_file_path,caffe_model_file_path,test_lmdb_path,meanB=128,meanG=128,meanR=128)


def get_traintest_from_proto(proto):
    logging.info('looking for traintest in '+proto)
    with open(proto,'r') as fp:
        train = None
        test = None
        traintest = None
        for line in fp:
            line = line.replace(' ','')  #line with spaces removed
            logging.debug('looking at line:'+line)
            if 'train_net:' in line and line[0] is not '#':
                train = line.replace('train_net:','').replace('"','').replace('\n','')
                print('train:'+train)
            if 'test_net:' in line and line[0] is not '#':
                test = line.replace('test_net:','').replace('"','').replace('\n','')
                print('test:'+test)
            if 'net:' in line and not 'test' in line and not 'train' in line and line[0] is not '#':
                traintest = line.replace('net:','').replace('"','').replace('\n','')
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


def get_labelfile_from_traintest(tfile,train_test='both'):
    print('looking for '+train_test+' in '+tfile)
    with open(tfile,'r') as fp:
        for line in fp:
            line = line.replace(' ','')  #line with spaces removed
            print('looking at line:'+line)
            if 'images_and_labels_file:' in line and line[0] is not '#':
                train = line.replace('train_net:','').replace('"','')
                print('train:'+train)
            if 'test_net:' in line and line[0] is not '#':
                test = line.replace('test_net:','').replace('"','')
                print('test:'+test)
            if 'net:' in line and not 'test' in line and not 'train' in line and line[0] is not '#':
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