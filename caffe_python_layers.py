import copy
import os
import caffe
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
import numpy as np
from PIL import Image
import cv2
import random
import string
import time
import multiprocessing
import lmdb
from multiprocessing.pool import ThreadPool as TPool

from trendi.utils import augment_images
from trendi.utils import imutils
from trendi import constants

class JrPixlevel(caffe.Layer):
    """
    loads images and masks for use with pixel level segmentation nets
    does augmentation on the fly
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        example
        layer {
            name: "data"
            type: "Python"
            top: "data"
            top: "label"
            python_param {
            module: "jrlayers"
            layer: "JrLayer"
            param_str: "{\'images_and_labels_file\': \'train_u21_256x256.txt\', \'mean\': (104.00699, 116.66877, 122.67892)}"
            }
        """
        # config
        params = eval(self.param_str)
        self.images_and_labels_file = params['images_and_labels_file']
        self.mean = np.array(params['mean'])
        self.random_init = params.get('random_initialization', True) #start from random point in image list
        self.random_pick = params.get('random_pick', True) #pick random image from list every time
        self.seed = params.get('seed', 1337)
        self.batch_size = params.get('batch_size',1)  #######Not implemented, batchsize = 1
        self.resize = params.get('resize',False)
        self.save_visual_output = params.get('save_visual_output',False)
        self.augment_images = params.get('augment',False)
        self.augment_max_angle = params.get('augment_max_angle',10)
        self.augment_max_offset_x = params.get('augment_max_offset_x',20)
        self.augment_max_offset_y = params.get('augment_max_offset_y',20)
        self.augment_max_scale = params.get('augment_max_scale',1.3)
        self.augment_max_noise_level = params.get('augment_max_noise_level',0)
        self.augment_max_blur = params.get('augment_max_blur',0)
        self.augment_do_mirror_lr = params.get('augment_do_mirror_lr',True)
        self.augment_do_mirror_ud = params.get('augment_do_mirror_ud',False)
        self.augment_crop_size = params.get('augment_crop_size',(224,224)) #
        self.augment_show_visual_output = params.get('augment_show_visual_output',False)
        self.augment_save_visual_output = params.get('augment_save_visual_output',False)
        self.augment_distribution = params.get('augment_distribution','uniform')
        self.n_labels = params.get('n_labels',16)
        self.fwd_pass_counter = 0
        self.images_processed_counter = 0
        self.start_time = time.time()
        self.analysis_time = time.time()
        self.analysis_time_out = time.time()
        self.cpu_count = multiprocessing.cpu_count()
        self.save_dir='./image_saves/'
        print('##############')
        print('params coming into jrlayers2')
        print('batchsize {}'.format(self.batch_size))
        print('imfile {} \nmean {}  \nrandinit {} \nrandpick {}'.format(self.images_and_labels_file, self.mean,self.random_init, self.random_pick))
        print('seed {} \nresize {} \nbatchsize {} \naugment {} \naugmaxangle {}'.format(self.seed,self.resize,self.batch_size,self.augment_images,self.augment_max_angle))
        print('augmaxdx {} \naugmaxdy {} \naugmaxscale {} \naugmaxnoise {} \naugmaxblur {}'.format(self.augment_max_offset_x,self.augment_max_offset_y,self.augment_max_scale,self.augment_max_noise_level,self.augment_max_blur))
        print('augmirrorlr {} \naugmirrorud {} \naugcrop {} \naugvis {}'.format(self.augment_do_mirror_lr,self.augment_do_mirror_ud,self.augment_crop_size,self.augment_show_visual_output))
        print('##############')
        print('waiting for you to read the above')
        t0=time.time()
        while(time.time()-t0 < 2 ): #to allow keypress disable whihc seems harder than i thought it was going to be
            time.sleep(0.1)
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        #if file not found and its not a path then tack on the training dir as a default locaiton for the trainingimages file
        if self.images_and_labels_file is not None:
            if os.path.isfile(self.images_and_labels_file):
                print('opening images_and_labelsfile '+str(self.images_and_labels_file))
                lines = open(self.images_and_labels_file, 'r').read().splitlines()
                self.imagefiles = [s.split()[0] for s in lines]
                self.labelfiles = [s.split()[1] for s in lines]
                self.n_files = len(self.imagefiles)
            else:
                logging.debug('COULD NOT OPEN  '+self.images_and_labels_file)
                print('COULD NOT OPEN  '+self.images_and_labels_file)
                return

#######begin vestigial code for separate images/labels files
        elif self.imagesfile is not None:
    #        if not os.path.isfile(self.imagesfile) and not '/' in self.imagesfile:
    #            self.imagesfile = os.path.join(self.images_dir,self.imagesfile)
            if not os.path.isfile(self.imagesfile):
                logging.warning('COULD NOT OPEN IMAGES FILE '+str(self.imagesfile))
            self.imagefiles = open(self.imagesfile, 'r').read().splitlines()
            self.n_files = len(self.imagefiles)

        elif self.labelsfile is not None:  #if labels flie is none then get labels from images
            if not os.path.isfile(self.labelsfile) and not '/' in self.labelsfile:
                self.labelsfile = os.path.join(self.labels_dir,self.labelsfile)
            if not os.path.isfile(self.labelsfile):
                print('COULD NOT OPEN labelS FILE '+str(self.labelsfile))
                self.labelfiles = open(self.labelsfile, 'r').read().splitlines()
###########end vestigial code

        print('found {} imagefiles and {} labelfiles'.format(len(self.imagefiles),len(self.labelfiles)))

        self.idx = 0
        # randomization: seed and pick
        if self.random_init:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.imagefiles)-1)
        logging.debug('initial self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

        ##check that all images are openable and have labels
        check_files = False
        if(check_files):
            good_img_files = []
            good_label_files = []
            print('checking image files')
            for ind in range(len(self.imagefiles)):
                img_arr = cv2.imread(self.imagefiles[ind]) #would be simpler to ditch that func and use cv2.imread, its never otherwise used
                if img_arr is not None:
                    label_arr = cv2.imread(self.labelfiles[ind])
                    if label_arr is not None:
                        #shapes match bet image and  mask
                        if label_arr.shape[0:2] == img_arr.shape[0:2]:  #the last dim is # channels (3 for img and dne for label
                            print('match index {} names {} {} imagesize {} and labelsize {}'.format(ind,self.imagefiles[ind],self.labelfiles[ind],img_arr.shape,label_arr.shape))
                            good_img_files.append(self.imagefiles[ind])
                            good_label_files.append(self.labelfiles[ind])
                            #image is big enough to crop out the min imsize for net (could be dealt with by resizing....)
                            # if label_arr.shape[0] >= self.augment_crop_size[0] and label_arr.shape[1] >= self.augment_crop_size[1]:
                            #     print('match index {} names {} {} imagesize {} and labelsize {}'.format(ind,self.imagefiles[ind],self.labelfiles[ind],img_arr.shape,label_arr.shape))
                            #     good_img_files.append(self.imagefiles[ind])
                            #     good_label_files.append(self.labelfiles[ind])
                            # else:
                            #     print('image too small ind {} names {} {} imsize {} lsize {}'.format(ind,self.imagefiles[ind],self.labelfiles[ind],img_arr.shape,label_arr.shape))
                        else:
                             print('shapes dont match ind {} names {} {} imsize {} lsize {}'.format(ind,self.imagefiles[ind],self.labelfiles[ind],img_arr.shape,label_arr.shape))
                else:
                    print('got bad image:'+self.imagefiles[ind])
            self.imagefiles = good_img_files
            self.labelfiles = good_label_files
            assert(len(self.imagefiles) == len(self.labelfiles))
            print('{} images and {} labels'.format(len(self.imagefiles),len(self.labelfiles)))
            self.n_files = len(self.imagefiles)
#            print(str(self.n_files)+' good files in image dir '+str(self.images_dir))

    def reshape(self, bottom, top):
        start_time=time.time()
   #     print('reshaping')
        # reshape tops to fit (leading 1 is for batch dimension)
#        self.data,self.label = self.load_image_and_mask()
        if self.batch_size == 1:
            self.data, self.label = self.load_image_and_mask()
        #add extra batch dimension
            top[0].reshape(1, *self.data.shape)
            top[1].reshape(1, *self.label.shape)
            logging.debug('batchsize 1 datasize {} labelsize {} '.format(self.data.shape,self.label.shape))
        else:
            all_data = np.zeros((self.batch_size,3,self.augment_crop_size[0],self.augment_crop_size[1]))      #Batchsizex3channelsxWxH
            all_labels = np.zeros((self.batch_size,1, self.augment_crop_size[0],self.augment_crop_size[1]) )
            multiprocess=False  ###DO THIS!!!!!  running out of threads...
            if multiprocess:
#                pool = multiprocessing.Pool(self.cpu_count)
                pool = TPool(self.cpu_count)
                output = pool.map(self.load_image_and_mask_helper,  range(self.batch_size))
#                output = pool.map(load_image_and_mask_helper2, [self for i in range(self.batch_size)])
                #hits cPickle.PicklingError: Can't pickle <type 'instancemethod'>: attribute lookup __builtin__.instancemethod failed
                i =0
                for o in output:
                    self.next_idx()  #this may only work if using random indexing not sequential...check that different images are getting sent to batch
                    all_data[i,...]=o[0]
                    all_labels[i,...]=o[1]
                    i=i+1
            else:
                for i in range(self.batch_size):
                    data, label = self.load_image_and_mask()
                    all_data[i,...]=data
                    all_labels[i,...]=label
                    self.next_idx()
            self.data = all_data
            self.label = all_labels
            #no extra dimension needed
            top[0].reshape(*self.data.shape)
            top[1].reshape(*self.label.shape)
            logging.debug('batchsize {} datasize {} labelsize {}'.format(self.batch_size,self.data.shape,self.label.shape))
        elapsed=time.time()-start_time
        print('reshape elapsed time:'+str(elapsed))

    def next_idx(self):
        if self.random_pick:
            self.idx = random.randint(0, len(self.imagefiles)-1)
        else:
            self.idx += 1
            if self.idx == len(self.imagefiles):
                self.idx = 0

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        # pick next input
        self.fwd_pass_counter += 1
        self.next_idx()


    def backward(self, top, propagate_down, bottom):
        pass

    def determine_label_filename(self,idx):
        if self.labelsfile is not None:
                filename = self.labelfiles[idx]
        #if there is no labelsfile specified then rename imagefiles to make labelfile names
        #so strip imagefile to get labelfile name
        else:
            filename = self.imagefiles[idx]
            filename = filename.split('.jpg')[0]
            filename = filename+self.labelfile_suffix

#        full_filename=os.path.join(self.labels_dir,filename)
#        return full_filename
        return filename

#     def load_image(self,idx):
#         """
#         Load input image and preprocess for Caffe:
#         - cast to float
#         - switch channels RGB -> BGR
#         - subtract mean
#         - transpose to channel x height x width order
#         """
#         while(1):
#             filename = self.imagefiles[idx]
# #            full_filename=os.path.join(self.images_dir,filename)
# #            print('the imagefile:'+full_filename+' index:'+str(idx))
#             label_filename=self.determine_label_filename(idx)
#             if not(os.path.isfile(label_filename) and os.path.isfile(filename)):
#                 print('ONE OF THESE IS NOT A FILE:'+str(label_filename)+','+str(filename))
#                 self.next_idx()
#             else:
#                 break
#         im = Image.open(filename)
#         if self.new_size:
#             im = im.resize(self.new_size,Image.ANTIALIAS)
#
#         in_ = np.array(im, dtype=np.float32)
#         if in_ is None:
#             logging.warning('could not get image '+full_filename)
#             return None
# #        print(full_filename+ ' has dims '+str(in_.shape))
#         in_ = in_[:,:,::-1]  #rgb->bgr
# #        in_ -= self.mean
#         in_ = in_.transpose((2,0,1))   #hwc->cwh
# #	print('uniques of img:'+str(np.unique(in_))+' shape:'+str(in_.shape))
#         return in_

#     def load_label_image(self, idx):
#         """
#         Load label image as 1 x height x width integer array of label indices.
#         The leading singleton dimension is required by the loss.
#         """
#         full_filename = self.determine_label_filename(idx)
#         im = Image.open(full_filename)
#         if im is None:
#             print(' COULD NOT LOAD FILE '+full_filename)
#             logging.warning('couldnt load file '+full_filename)
#         if self.new_size:
#             im = im.resize(self.new_size,Image.ANTIALIAS)
#
#         in_ = np.array(im, dtype=np.uint8)
#
#         if len(in_.shape) == 3:
# #            logging.warning('got 3 layer img as mask, taking first layer')
#             in_ = in_[:,:,0]
#     #        in_ = in_ - 1
#  #       print('uniques of label:'+str(np.unique(in_))+' shape:'+str(in_.shape))
#  #       print(full_filename+' has dims '+str(in_.shape))
#         label = copy.copy(in_[np.newaxis, ...])
# #        print('after extradim shape:'+str(label.shape))
#         return label

    def load_image_and_mask_helper(self,idxs):
        #this is to alow multiprocess to send an unneeded argument , probably there is some way to multiprocess without args
        #without this hack but this works and is easy
        out1,out2=self.load_image_and_mask()
        return out1,out2

    def load_image_and_mask(self):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #add idx and allow randomness of next_idx to take care of multiprocessing case where
        #multiple threads are using this simultanesouly and dont want to get the same image
        self.analysis_time=time.time()
        self.next_idx()
        while(1):
            filename = self.imagefiles[self.idx]
            label_filename=self.labelfiles[self.idx]
            print('imagefile:'+filename+'\nlabelfile:'+label_filename+' index:'+str(self.idx))
            if not(os.path.isfile(label_filename) and os.path.isfile(filename)):
                print('ONE OF THESE IS NOT ACCESSIBLE:'+str(label_filename)+','+str(filename))
                self.next_idx()
                continue
                ####todo - check that the image is coming in correctly wrt color etc
#            im = Image.open(filename)
            im = cv2.imread(filename)
            if im is None:
                logging.warning('could not get image1 '+filename)
                self.next_idx()
                continue

            if self.resize:
#                im = im.resize(self.resize,Image.ANTIALIAS)
                logging.debug('resizing image, start shape {} '.format(im.shape))
                im = imutils.resize_keep_aspect(im,output_size=self.resize,careful_with_the_labels=False)
                logging.debug('resized image, end shape {} '.format(im.shape))
            in_ = np.array(im, dtype=np.float32)
#            in_ = in_[:,:,::-1]   #RGB -> BGR no need since using cv2.imread which gives bgr
            if in_ is None:
                logging.warning('could not get image2 '+filename)
                self.next_idx()
                continue
            """
            Load label image as 1 x height x width integer array of label indices.
            The leading singleton dimension is required by the loss.
            """
#            im = Image.open(label_filename)
            im = cv2.imread(label_filename)
            if im is None:
                logging.warning('could not get label1 '+filename)
                self.next_idx() #
                continue
            if self.resize:
                #this should be done with imutils.resize_keep_aspect(...careful_with_the_labels=True), no ???
                logging.debug('resizing mask, start shape {} '.format(im.shape))
                im = imutils.resize_keep_aspect(im,output_size=self.resize,careful_with_the_labels=True)
                logging.debug('resized mask, end shape {} '.format(im.shape))
#                im = im.resize(self.resize,Image.ANTIALIAS)
            if im is None:
                logging.warning('couldnt load label '+label_filename)
                self.next_idx()
                continue
    #        if self.new_size:
    #            im = im.resize(self.new_size,Image.ANTIALIAS)
            label_in_ = np.array(im, dtype=np.uint8)
            if in_ is None:
                logging.warning('could not get image '+filename)
                self.next_idx()
                continue
            break  #we finally made it past all the checks
        # if self.kaggle is not False:
        #     print('kagle image, moving 255 -> 1')
        #     label_in_[label_in_==255] = 1
#        in_ = in_ - 1
 #       print('uniques of label:'+str(np.unique(label_in_))+' shape:'+str(label_in_.shape))
#        print('after extradim shape:'+str(label.shape))
#        out1,out2 = augment_images.generate_image_onthefly(in_, mask_filename_or_nparray=label_in_)
        logging.debug('img/mask sizes in jrlayers2: {} and {}, cropsize {} maxangle {}'.format(in_.shape,label_in_.shape,self.augment_crop_size,self.augment_max_angle))
#        print('img/mask sizes in jrlayers2: {} and {}, cropsize {} angle {}'.format(in_.shape,label_in_.shape,self.augment_crop_size,self.augment_max_angle))

        logging.debug('uniques b4 '+str(np.unique(label_in_)))


        if self.save_visual_output:
            lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(30)]
            name = os.path.join(self.save_dir,"".join(lst))
            cv2.imwrite(name+'.jpg',in_)
            maskname = name+'_mask.png'
            cv2.imwrite(maskname,label_in_)
            print('writing images to {} and {}'.format(name+'.jpg',maskname))
            imutils.show_mask_with_labels(maskname,labels=constants.pixlevel_categories_v3,original_image=name+'.jpg',visual_output=False,savename=name+'_check_legend.jpg',save_images=True)

        out1, out2 = augment_images.generate_image_onthefly(in_, mask_filename_or_nparray=label_in_,
            gaussian_or_uniform_distributions=self.augment_distribution,
            max_angle = self.augment_max_angle,
            max_offset_x = self.augment_max_offset_x,max_offset_y = self.augment_max_offset_y,
            max_scale=self.augment_max_scale,
            max_noise_level=self.augment_max_noise_level,noise_type='gauss',
            max_blur=self.augment_max_blur,
            do_mirror_lr=self.augment_do_mirror_lr,
            do_mirror_ud=self.augment_do_mirror_ud,
            crop_size=self.augment_crop_size,
            show_visual_output=self.augment_show_visual_output)

        if self.save_visual_output:
            name = name+"after_aug"
            cv2.imwrite(name+'.jpg',out1)
            maskname = name+'_mask.png'
            cv2.imwrite(maskname,out2)
            print('writing images to {} and {}'.format(name+'.jpg',maskname))
            imutils.show_mask_with_labels(maskname,labels=constants.pixlevel_categories_v3,original_image=name+'.jpg',visual_output=False,savename=name+'_check_legend.jpg',save_images=True)

        logging.debug('uniques after '+str(np.unique(out2)))
#        out1 = out1[:,:,::-1]   #RGB -> BGR - not necesary since this is done above (line 303)
        ('out1 shape {} type {} 2 shape {} type {}'.format(out1.shape,type(out1),out2.shape,type(out2)))
        out1 -= self.mean  #assumes means are BGR order, not RGB
        out1 = out1.transpose((2,0,1))  #wxhxc -> cxwxh
        if len(out2.shape) == 3:
            logging.warning('got 3 layer img as mask from augment, taking first layer')
            out2 = out2[:,:,0]
        out2 = copy.copy(out2[np.newaxis, ...])

        dt_in = time.time()-self.analysis_time
        dt_out = self.analysis_time - self.analysis_time_out
        dt_tot = time.time() - self.analysis_time_out
        total_elapsed_time = time.time() - self.start_time
        self.analysis_time_out = time.time()
        self.images_processed_counter += 1

        print(str(self.fwd_pass_counter)+' fwd passes, '+
              str(self.images_processed_counter)+' images processed.' +
              ' tout '+str(round(dt_out,3))+
              ' tin '+str(round(dt_in,3))+
              ' ttot '+str(round(dt_tot,3))+
              ' tglob '+str(round(total_elapsed_time,3))+
              ' tpi '+str(round(float(total_elapsed_time)/self.images_processed_counter,3)))
        return out1,out2



def load_image_and_mask_helper2(pixobj):
    #this is to alow multiprocess to send an unneeded argument , probably there is some way to multiprocess without args
    #without this hack but this works and is easy
    out1,out2=pixobj.load_image_and_mask()
    return out1,out2

    def squared_cubed(self,x):
        return x**2,x**3

    def noargs(self):
        return 3**2,3**3

    def helpnoargs(self,X):
        r=noargs()
        return r

    def test_multi(self,bsize):
    #    bsize = 4
        pool = multiprocessing.Pool(20)
        ins = range(bsize)
    #    outs = zip(*pool.map(squared, range(bsize)))
        outs = pool.map(self.squared_cubed, ins)
        print outs
        outs = pool.map(self.helpnoargs, ins)
        print outs
        theout1=[]
        theout2=[]
        for o in outs:
            theout1.append(o[0])
            theout2.append(o[1])
        print theout1
        print theout2






























######################################################################################3
# SINGLE/MULTILABEL
#######################################################################################

class JrMultilabel(caffe.Layer):
    """
    Load (input image, label vector) pairs where label vector is like [0 1 0 0 0 1 ... ]

    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        example
        layer {
            name: "data"
            type: "Python"
            top: "data"
            top: "label"
            python_param {
            module: "jrlayers"
            layer: "JrMultilabel"
            param_str: "{\'images_and_labels_file\': \'/home/jeremy/image_dbs/tamara_berg/web1\', \'mean\': (104.00699, 116.66877, 122.67892)}"
            }
        """
        # config
        params = eval(self.param_str)
#mandatory argument
#        self.parname = params['param_name']
#optional argument
 #       self.parname = params.get('param_name',default_value)

        self.images_and_labels_file = params.get('images_and_labels_file',None)
        self.lmdb = params.get('lmdb',None)
        self.mean = np.array(params.get('mean',(0,0,0)))
        self.random_init = params.get('random_initialization', True) #start from random point in image list
        self.random_pick = params.get('random_pick', True) #pick random image from list every time
        self.seed = params.get('seed', 1337)
        self.new_size = params.get('resize',None)
        self.batch_size = params.get('batch_size',1)
        self.regression = params.get('regression',False)
        self.scale = params.get('scale',False)
        self.save_visual_output = params.get('save_visual_output',False)
        self.equalize_category_populations = params.get('equalize_categories',False)
        self.max_category_index = params.get('max_category_index',True)
        self.augment_images = params.get('augment',False)
        self.augment_max_angle = params.get('augment_max_angle',10)
        self.augment_max_offset_x = params.get('augment_max_offset_x',20)
        self.augment_max_offset_y = params.get('augment_max_offset_y',20)
        self.augment_max_scale = params.get('augment_max_scale',1.4)
        self.augment_max_noise_level = params.get('augment_max_noise_level',0)
        self.augment_max_blur = params.get('augment_max_blur',0)
        self.augment_do_mirror_lr = params.get('augment_do_mirror_lr',True)
        self.augment_do_mirror_ud = params.get('augment_do_mirror_ud',False)
        self.augment_crop_size = params.get('augment_crop_size',(224,224)) #
        self.augment_show_visual_output = params.get('augment_show_visual_output',False)
        self.augment_save_visual_output = params.get('augment_save_visual_output',False)
        self.augment_distribution = params.get('augment_distribution','uniform')
        self.n_labels = params.get('n_labels',0)  #this will obvious from the image/label file. in case of multilabel this is number of classes, i n case of single label this is 1
        self.counter = 0


        print('############net params for jrlayers2#########')
        if self.images_and_labels_file is not None:
            print('im/label file {}'.format(self.images_and_labels_file))
        if self.lmdb is not None:
            print('lmdb {}'.format(self.lmdb))
        print('mean {}  randinit {} randpick {} '.format( self.mean,self.random_init, self.random_pick))
        print('seed {} newsize {} batchsize {} augment {} augmaxangle {} '.format(self.seed,self.new_size,self.batch_size,self.augment_images,self.augment_max_angle))
        print('augmaxdx {} augmaxdy {} augmaxscale {} augmaxnoise {} augmaxblur {} '.format(self.augment_max_offset_x,self.augment_max_offset_y,self.augment_max_scale,self.augment_max_noise_level,self.augment_max_blur))
        print('augmirrorlr {} augmirrorud {} augcrop {} augvis {}'.format(self.augment_do_mirror_lr,self.augment_do_mirror_ud,self.augment_crop_size,self.augment_show_visual_output))
        print('scale {}'.format(self.scale))
        print('############end of net params for jrlayers2#########')

        self.idx = 0
        self.images_processed = 0
        self.analysis_time = time.time()
        self.analysis_time_out = time.time()
        self.previous_images_processed=0
        self.default_newsize=(256,256)
        # print('images+labelsfile {} mean {}'.format(self.images_and_labels_file,self.mean))
        # two tops: data and label
        if len(top) != 2:
            print('len of top is '+str(len(top)))
#            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
#
        # load indices for images and labels
        #if file not found and its not a path then tack on the training dir as a default locaiton for the trainingimages file
        if self.images_and_labels_file is not None:
            print('using images/labels file '+self.images_and_labels_file)
 #           if not os.path.isfile(self.images_and_labels_file) and not '/' in self.images_and_labels_file:
 #               if self.images_dir is not None:
 #                   self.images_and_labels_file = os.path.join(self.images_dir,self.images_and_labels_file)
            if not os.path.isfile(self.images_and_labels_file):
                print('COULD NOT OPEN IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                logging.debug('COULD NOT OPEN IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                sys.exit('COULD NOT OPEN IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                return
            self.images_and_labels_list = open(self.images_and_labels_file, 'r').read().splitlines()
   #         print('imgs:'+str(self.images_and_labels_list))
            if self.images_and_labels_list is None or len(self.images_and_labels_list)==0:
                print('COULD NOT FIND ANYTHING IN  IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                logging.debug('COULD NOT FIND ANYTHING IN IMAGES/LABELS FILE '+str(self.images_and_labels_file))
                return
            self.n_files = len(self.images_and_labels_list)
            logging.debug('images and labels file: {} n: {}'.format(self.images_and_labels_file,self.n_files))
    #        self.indices = open(split_f, 'r').read().splitlines()
    #build list of files
            good_img_files = []
            good_label_vecs = []
            max_cat_index=0
            for line in self.images_and_labels_list:
                imgfilename = line.split()[0]
                vals = line.split()[1:]
                self.n_labels = len(vals)
                if self.regression:
                    label_vec = [float(i) for i in vals]
                else:
                    try:
                        label_vec = [int(i) for i in vals]
                    except:
                        logging.debug('got something that coulndt be turned into a string in the following line from file '+self.images_and_labels_file)
                        logging.debug(line)
                        logging.debug('error:'+str(sys.exc_info()[0])+' , skipping line')
                        continue
                label_vec = np.array(label_vec)
                max_cat_index=np.max([max_cat_index,np.max(label_vec)]) #abandoning automatic calc of max cat index since also have to do it for lmdb
                self.n_labels = len(label_vec)
                if self.n_labels == 1:
  #                  print('length 1 label')
                    label_vec = label_vec[0]
                good_img_files.append(imgfilename)
                good_label_vecs.append(label_vec)
            self.imagefiles = good_img_files
            self.label_vecs = good_label_vecs

            check_files = False
            ##check that all images are openable and have labels
            if check_files:
                print('checking image files')
                for line in self.images_and_labels_list:
                    imgfilename = line.split()[0]
                    # img_arr = Image.open(imgfilename)
                    img_arr = cv2.imread(imgfilename)
                    in_ = np.array(img_arr, dtype=np.float32)
                    if img_arr is not None:
                        vals = line.split()[1:]
                        label_vec = [int(i) for i in vals]
                        label_vec = np.array(label_vec)
                        self.n_labels = len(label_vec)   #the length of the label vector for multiclass data
                        if self.n_labels == 1:  #for the case of single_class data
                            label_vec = label_vec[0]    #                label_vec = label_vec[np.newaxis,...]  #this is required by loss whihc otherwise throws:
                        if label_vec is not None:
                            if len(label_vec) > 0:  #got a vec
                                good_img_files.append(imgfilename)
                                good_label_vecs.append(label_vec)
                                sys.stdout.write(spinner.next())
                                sys.stdout.flush()
                                sys.stdout.write('\b')
                          #      print('got good image of size {} and label of size {}'.format(in_.shape,label_vec.shape))
                            else:
                                print('something wrong w. image of size {} and label of size {}'.format(in_.shape,label_vec.shape))
                    else:
                        print('got bad image:'+line)

            assert(len(self.imagefiles) == len(self.label_vecs))
            #print('{} images and {} labels'.format(len(self.imagefiles),len(self.label_vecs)))
            self.n_files = len(self.imagefiles)
            print(str(self.n_files)+' good files found in '+self.images_and_labels_file)

    #use lmdb
        elif self.lmdb is not None:
            self.lmdb_env = lmdb.open(self.lmdb, readonly=True)
#            with self.lmdb_env.begin() as self.txn:
            self.txn = self.lmdb_env.begin() #can also do begin(buffers=True), check https://lmdb.readthedocs.io/en/release/ for wtf this does

            self.n_files = self.txn.stat()['entries']
            print('lmdb {} opened\nstat {}\nentries {}'.format(self.lmdb,self.txn.stat(),self.n_files))
        #get first dB entry to determine label size
            str_id = '{:08}'.format(0)
            raw_datum = self.txn.get(str_id.encode('ascii'))
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = np.fromstring(datum.data, dtype=np.uint8)
            y = datum.label
#                vals = y.split() #in the meantime lmdb cant handle multilabel
            self.n_labels = 1
            print('lmdb label {} length {} datashape {}'.format(y,self.n_labels,flat_x.shape))
            #populate label_vecs to allow even distribution of examples
            self.label_vecs = []
            for dummy in range(self.n_files):
                try:
                    str_id = '{:08}'.format(dummy)
                    raw_datum = self.txn.get(str_id.encode('ascii'))
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.ParseFromString(raw_datum)
                    y = datum.label
                    self.label_vecs.append(y)
                except:
                    print('error getting record {} from db'.format(dummy))
                    break

        self.idx = 0
        # randomization: seed and pick
        if self.random_init:
            random.seed(self.seed)
            self.idx = random.randint(0, self.n_files-1)
#        if self.random_pick:
#            random.shuffle(self.images_and_labels_list)
        print('initial self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

        spinner = spinning_cursor()
        print('self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))

        #size_for_shaping is the actual final image size. Image gets resized to new_size if it exists, and cropped
        #to augment_crop_size if that exists. So size_for_shaping = augment_cropsize if that exists, otherwise new_size
        if self.augment_crop_size is not None and self.augment_images is True:
            top[0].reshape(self.batch_size, 3,self.augment_crop_size[0], self.augment_crop_size[1])
            self.size_for_shaping = self.augment_crop_size
            print('dba')
            if self.new_size is None:
                logging.warning('WARNING!!! got no size for self.newsize, using '+str(self.default_newsize)+' resize and and  crop '+str(self.augment_crop_size))
             #   raw_input('ret to cont')
                self.new_size=self.default_newsize
        elif self.new_size is not None:
            top[0].reshape(self.batch_size, 3, self.new_size[0], self.new_size[1])
            self.size_for_shaping = self.new_size
            print('dbb')
        else:
            logging.warning('WARNING!!! got no crop or size for self.newsize, using 224x224 resize and no crop!!')
          #  raw_input('ret to cont')
            self.new_size = (224,224)
            top[0].reshape(self.batch_size, 3, self.new_size[0], self.new_size[1])
            self.size_for_shaping = (224,224)
            print('dbc')
        print('size for shaping (final img size):'+str(self.size_for_shaping))
        top[1].reshape(self.batch_size, self.n_labels)

        #EQUALIZE CATEGORY POPULATIONS STUFF
        #get examples into distinct lists one for each category
        #self.label_vecs is the categories in ordered list by idx
        #so convert that to several lists of idx's, one per category
        if self.equalize_category_populations != False:
            self.idx_per_cat = {}
            for idx in range(self.n_files):
                label = self.label_vecs[idx]
                if not label in self.idx_per_cat:
                    self.idx_per_cat[label]=[idx]
                else:
                    self.idx_per_cat[label].append(idx)
            self.idx_per_cat_lengths = [len(self.idx_per_cat[k]) for k in self.idx_per_cat]
#            raw_input('ret to cont')
            self.n_seen_per_category = np.zeros(self.max_category_index)
            self.max_category_index = max([k for k in self.idx_per_cat])
#            print('pops:'+str(self.idx_per_cat)+' max cat index:'+str(self.max_category_index))

            if self.equalize_category_populations == True:
                self.category_population_percentages = [1.0/(self.max_category_index+1) for i in range(self.max_category_index+1)]
            else:  #user explicitly gave list of desired percentages
                self.category_population_percentages = self.equalize_category_populations
            #done - add files per class (from create_nn_imagelsts)
            print('desired population percentages:'+str(self.category_population_percentages))
            #populations - the initial 1 below is a white lie (they really start at 0 of course) but this way I avoid divide-by-0 on first run without checking every time
            self.category_populations_seen = [1 for dummy in range(self.max_category_index+1)]
            self.worst_off = 0
        time.sleep(2) #give some time to read how many imgs in the labelfile.

        self.start_time=time.time()


    def reshape(self, bottom, top):
        pass
        #print('start reshape')
#        logging.debug('self.idx is :'+str(self.idx)+' type:'+str(type(self.idx)))
        if self.batch_size == 1:
            if self.lmdb is not None:
                imgfilename, self.data, self.label = self.load_image_and_label_from_lmdb()
            else:
                imgfilename, self.data, self.label = self.load_image_and_label()
            self.images_processed += 1
        else:

            all_data = np.zeros((self.batch_size,3,self.size_for_shaping[0],self.size_for_shaping[1]))
            all_labels = np.zeros((self.batch_size,self.n_labels))
            for i in range(self.batch_size):
                if self.lmdb is not None:
                    imgfilename, data, label = self.load_image_and_label_from_lmdb()
                else:
                    imgfilename, data, label = self.load_image_and_label()
                all_data[i,...]=data
                all_labels[i,...]=label
                self.next_idx()
            self.data = all_data
            self.label = all_labels
            self.previous_images_processed = self.images_processed
            self.images_processed += self.batch_size


        ## reshape tops to fit (leading 1 is for batch dimension)
 #       top[0].reshape(1, *self.data.shape)
 #       top[1].reshape(1, *self.label.shape)
 #        print('top 0 shape {} top 1 shape {}'.format(top[0].shape,top[1].shape))
 #        print('data shape {} label shape {}'.format(self.data.shape,self.label.shape))
##       the above just shows objects , top[0].shape is an object apparently

    def next_idx(self):
        if self.equalize_category_populations:
            actual_fractions_seen = np.divide([float(dummy) for dummy in self.category_populations_seen],
                                              np.sum(self.category_populations_seen))
            diff = self.category_population_percentages - actual_fractions_seen
            self.worst_off = np.argmax(diff)
#            print('desired {}\nmost distant {}\ndiff {}\nactual {}'.format(self.category_population_percentages,self.worst_off,diff,
#                                            actual_fractions_seen))
            print('populations seen: {}'.format(self.category_populations_seen))
            n_examples = len(self.idx_per_cat[self.worst_off])
            self.idx = self.idx_per_cat[self.worst_off][np.random.randint(0,n_examples)]
            #raw_input('idx: {} ret to cont'.format(self.idx))
        elif self.random_pick:
#            self.idx = random.randint(0, len(self.imagefiles)-1)
            self.idx = random.randint(0, self.n_files-1)
            logging.debug('next idx='+str(self.idx))
        else:
            self.idx += 1
            if self.idx == len(self.imagefiles):
                print('hit end of labels, going back to first')
                self.idx = 0

    def forward(self, bottom, top):
        # assign output
        #print('forward start')
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        # pick next input
        self.next_idx()
        #print('forward end')
        self.counter += 1
   #     print('data shape {} labelshape {} label {} '.format(self.data.shape,self.label.shape,self.label))

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image_and_label(self,idx=None):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - do random xforms (rotate, translate, crop, possibly noise)
        - subtract mean
        - transpose to channel x height x width order
        """
        #print('load_image_and_label start')
        dt_tot = time.time() - self.analysis_time
        self.analysis_time=time.time()
        while(1):

            filename = self.imagefiles[self.idx]
            label_vec = self.label_vecs[self.idx]
 #           if self.images_dir:
 #               filename=os.path.join(self.images_dir,filename)
            print('the imagefile:'+filename+' label '+str(label_vec)+' index:'+str(idx))
            if not(os.path.isfile(filename)):
                print('NOT A FILE:'+str(filename)+' ; trying next')
                self.next_idx()   #bad file, goto next
                continue
            #print('calling augment_images with file '+filename)
#############start added code to avoid cv2.imread############
            try:
                # im = Image.open(filename)
                im = cv2.imread(filename)

                if im is None:
                    logging.warning('jrlayers2 could not get im '+filename)
                    self.next_idx()
                    continue

                in_ = np.array(im, dtype=np.float32)
                if self.new_size is not None and (in_.shape[0] != self.new_size[0] or in_.shape[1] != self.new_size[1]):
           #         im = im.resize(self.new_size,Image.ANTIALIAS)
                    print('resizing {} from {} to {}'.format(filename, in_.shape,self.new_size))
                  #  raw_input('ret to cont' )
                    in_ = imutils.resize_keep_aspect(in_,output_size=self.new_size)
##                     print('new shape '+str(in_.shape))

                if in_ is None:
                    logging.warning('jrlayers2 could not get in_ '+filename)
                    self.next_idx()
                    continue
                logging.debug('IN_ SHAPE in jrlayers2:'+str(in_.shape))
                if in_.shape[2] != 3:
                    logging.debug('got channels !=3 in jrlayers2.load_image_and_labels')
                    self.next_idx()
                    continue
            except:
                e = sys.exc_info()[0]
                logging.debug( "Error {} in jrlayers2 checking image {}".format(e,filename))
                self.next_idx()
                continue
            # try:
            #     in_ = in_[:,:,::-1]  #RGB->BGR - since we're using PIL Image to read in .  The caffe default is BGR so at inference time images are read in as BGR
            # except:
            #     e = sys.exc_info()[0]
            #     logging.debug( "Error in jrlayers2 transposing image rgb->bgr: %s" % e )
            #     self.next_idx()
            #     continue

#############end added code to avoid cv2.imread############

            out_ = augment_images.generate_image_onthefly(in_, gaussian_or_uniform_distributions=self.augment_distribution,
                max_angle = self.augment_max_angle,
                max_offset_x = self.augment_max_offset_x,max_offset_y = self.augment_max_offset_y,
                max_scale=self.augment_max_scale,
                max_noise_level=self.augment_max_noise_level,noise_type='gauss',
                max_blur=self.augment_max_blur,
                do_mirror_lr=self.augment_do_mirror_lr,
                do_mirror_ud=self.augment_do_mirror_ud,
                crop_size=self.augment_crop_size,
                show_visual_output=self.augment_show_visual_output,
                                save_visual_output=self.augment_save_visual_output)

#            out_,unused = augment_images.generate_image_onthefly(in_,mask_filename_or_nparray=in_)
#            out_ = augment_images.generate_image_onthefly(in_)

            #print('returned from augment_images')
            #im = Image.open(filename)
            #if im is None:
            #    logging.warning('could not get image '+filename)
            #    self.next_idx()
            #    idx = self.idx
            #    continue
            #if self.new_size:
            #    im = im.resize(self.new_size,Image.ANTIALIAS)
            if out_ is None:
                logging.warning('could not get image '+filename)
                self.next_idx()
                continue
            if len(out_.shape) != 3 :
                print('got strange-sized img not having 3 dimensions ('+str(out_.shape) + ') when expected shape is hxwxc (3 dimensions)')
                print('weird file:'+filename)
                self.next_idx()  #goto next
                continue

    #if there's a crop then check resultsize=cropsize.
            if self.augment_crop_size is not None and (out_.shape[0] != self.augment_crop_size[0] or out_.shape[1] != self.augment_crop_size[1]):
                    print('got strange-sized img of size '+str(out_.shape) + ' when expected cropped hxw is '+str(self.augment_crop_size))
                    print('weird file:'+filename)
                    self.next_idx()  #goto next
                    continue
    #If there's no crop but there is a resize, check resultsize=resize_size
            if self.augment_crop_size is None and self.new_size is not None and (out_.shape[0] != self.new_size[0] or out_.shape[1] != self.new_size[1]):
                    print('got strange-sized img of size '+str(out_.shape) + ' when expected resized hxw is '+str(self.new_size))
                    print('weird file:'+filename)
                    self.next_idx()  #goto next
                    continue

            if out_.shape[2] !=3 :
                print('got non-3-chan img of size '+str(out_.shape) + ' when expected n_channels is 3 '+str(self.new_size))
                print('weird file:'+filename)
                self.next_idx()  #goto next
                continue
            break #got good img after all that , get out of while

        if self.augment_save_visual_output:
            name = str(self.idx)+str(label_vec)+'jr.jpg'
            cv2.imwrite(name,out_)
            print('saving '+name)
        out_ = np.array(out_, dtype=np.float32)

        #print(str(filename) + ' has dims '+str(out_.shape)+' label:'+str(label_vec)+' idex'+str(idx))
        #todo maybe also normalize to -1:1
        out_ -= self.mean
        out_ = out_.transpose((2,0,1))  #Row Column Channel -> Channel Row Column
#	print('uniques of img:'+str(np.unique(in_))+' shape:'+str(in_.shape))
        #print('load_image_and_label end')
        if self.scale:
            if self.scale==True:
                out_=out_/255.0
            else:
                out_=out_/self.scale
        dN = self.images_processed - self.previous_images_processed
        dt_in = time.time()-self.analysis_time
        dt_out = time.time()-self.analysis_time_out
        total_elapsed_time = time.time() - self.start_time
        self.analysis_time_out = time.time()
        print(str(self.counter)+' fwd passes, '+str(self.images_processed)+
              ' images processed, dN/dt='+str(round(float(self.images_processed)/total_elapsed_time,3))+
              ' tin '+str(round(dt_in,3))+
              ' tout '+str(round(dt_out,3))+
              ' ttot '+str(round(dt_tot,3)))

        if self.equalize_category_populations:
#            cat_of_image_seen = self.idx
            self.category_populations_seen[self.worst_off]+=1

        return filename, out_, label_vec

    def load_image_and_label_from_lmdb(self,idx=None):
        """
        Load input image, label from lmdb and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - do random xforms (rotate, translate, crop, possibly noise)
        - subtract mean
        - transpose to channel x height x width order
        - note this currently only works with single-label info as the lmdb label is expected (by caffe)
          to be an int or long, so no way to cram a vector in there
        """
        logging.debug('load_image_and_label start, idx:'+str(self.idx))
        while(1):

            str_id = '{:08}'.format(self.idx)
            raw_datum = self.txn.get(str_id.encode('ascii'))
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw_datum)
            flat_x = np.fromstring(datum.data, dtype=np.uint8) #is this right, given that this may be neg and float numbers...maybe just save as un-normalized positive  uint8s to save space
            orig_x = flat_x.reshape(datum.channels, datum.height, datum.width)
            logging.debug('strid {} channels {} width {} height {} flatxsize {} reshaped {} min {} max {}'.format(str_id,datum.channels,datum.width,datum.height,len(flat_x),orig_x.shape,np.min(orig_x),np.max(orig_x)))
            if datum.channels == 3:
                logging.debug('before transpose shape:'+str(orig_x.shape))
            # as the input is transposed to c,h,w  by transpose(2,0,1) we have to undo it with transpose(1,2,0)
            #h w c  transpose(2,0,1) -> c h w
            #c h w  transpose(1,2,0) -> h w c
 #               transpose_time=time.time()
                x = orig_x.transpose((1,2,0)) #get hwc image
#                print('transposetime '+str(time.time()-transpose_time))
                logging.debug('after transpose shape:'+str(x.shape))
            elif datum.channels == 1:
#                 print('reshaping 1 chan')
                x = flat_x.reshape(datum.height, datum.width)
            y = datum.label
#            vals = y.split()
#            print('lmdb label {} length {}'.format(y,self.n_labels))
            if self.regression:
                label_vec = [float(i) for i in y] #currently not possible
            else:
                try:
                    label_vec = y
                except:
                    logging.debug('got something that coulndt be turned into a label from lmdb '+self.lmdb)
                    logging.debug('error:'+str(sys.exc_info()[0])+' , skipping line, index='+str(self.idx))
                    self.next_idx()
                    continue
            label_vec = np.array(label_vec)
            logging.debug('label vec:'+str(label_vec))
            in_ = np.array(x, dtype=np.float32)
            if self.new_size is not None and (in_.shape[0] != self.new_size[0] or in_.shape[1] != self.new_size[1]):
       #         im = im.resize(self.new_size,Image.ANTIALIAS)
                print('resizing from {} to {}'.format(in_.shape,self.new_size))
                in_ = imutils.resize_keep_aspect(in_,output_size=self.new_size)
##                     print('new shape '+str(in_.shape))
            if in_ is None:
                logging.warning('jrlayers2 could not get in_ of idx:'+str(self.idx))
                self.next_idx()
                continue
            logging.debug('IN_ SHAPE in jrlayers2:'+str(in_.shape))
            if in_.shape[2] != 3:
                logging.debug('got channels !=3 in jrlayers2.load_image_and_labels')
                self.next_idx()
                continue
#            try:
#                in_ = in_[:,:,::-1]  #RGB->BGR - since we're using PIL Image to read in .  The caffe default is BGR so at inference time images are read in as BGR
#            except:
#                e = sys.exc_info()[0]
#                logging.debug( "Error in jrlayers2 transposing image rgb->bgr: %s" % e )
#                self.next_idx()
#                continue

            if self.save_visual_output:
                name = str(self.idx)+str(label_vec)+'_before_aug.jpg'
                cv2.imwrite(name,in_)
                print('saving '+name)


            out_ = augment_images.generate_image_onthefly(in_, gaussian_or_uniform_distributions=self.augment_distribution,
                max_angle = self.augment_max_angle,
                max_offset_x = self.augment_max_offset_x,max_offset_y = self.augment_max_offset_y,
                max_scale=self.augment_max_scale,
                max_noise_level=self.augment_max_noise_level,noise_type='gauss',
                max_blur=self.augment_max_blur,
                do_mirror_lr=self.augment_do_mirror_lr,
                do_mirror_ud=self.augment_do_mirror_ud,
                crop_size=self.augment_crop_size,
                show_visual_output=self.augment_show_visual_output,
                                save_visual_output=self.augment_save_visual_output)
            if out_ is None:
                logging.warning('could not get augmented image idx'+str(self.idx))
                self.next_idx()
                continue
            if len(out_.shape) != 3 :
                print('got strange-sized img not having 3 dimensions ('+str(out_.shape) + ') when expected shape is hxwxc (3 dimensions)')
                print('weird file:'+str(self.idx))
                self.next_idx()  #goto next
                continue

    #if there's a crop then check resultsize=cropsize.
            if self.augment_crop_size is not None and (out_.shape[0] != self.augment_crop_size[0] or out_.shape[1] != self.augment_crop_size[1]):
                    print('got strange-sized img of size '+str(out_.shape) + ' when expected cropped hxw is '+str(self.augment_crop_size))
                    print('weird file , crop+resize idx:'+str(self.idx))
                    self.next_idx()  #goto next
                    continue
    #If there's no crop but there is a resize, check resultsize=resize_size
            if self.augment_crop_size is None and self.new_size is not None and (out_.shape[0] != self.new_size[0] or out_.shape[1] != self.new_size[1]):
                    print('got strange-sized img of size '+str(out_.shape) + ' when expected resized hxw is '+str(self.new_size))
                    print('weird file, no crop+resize idx:'+str(self.idx))
                    self.next_idx()  #goto next
                    continue

            if out_.shape[2] !=3 :
                print('got non-3-chan img of size '+str(out_.shape) + ' when expected n_channels is 3 '+str(self.new_size))
                print('weird file:'+str(self.idx))
                self.next_idx()  #goto next
                continue
            break #got good img after all that , get out of while

        if self.save_visual_output:
            name = str(self.idx)+str(label_vec)+'_after_aug.jpg'
            cv2.imwrite(name,out_)
            print('saving '+name)
        out_ = np.array(out_, dtype=np.float32)
        #print(str(filename) + ' has dims '+str(out_.shape)+' label:'+str(label_vec)+' idex'+str(idx))
        out_ -= self.mean
        out_ = out_.transpose((2,0,1))  #Row Column Channel -> Channel Row Column
        #DONE maybe also normalize to -1:1
        if self.scale:
            if self.scale==True:
                out_=out_/255.0
            else:
                out_=out_/self.scale
#	print('uniques of img:'+str(np.unique(in_))+' shape:'+str(in_.shape))
        #print('load_image_and_label end')
        return self.idx, out_, label_vec


def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor
















































######################################################################################3
# test
#######################################################################################

class JrTestInput(caffe.Layer):
    """
    Load (input image, label vector) pairs where label vector is like [0 1 0 0 0 1 ... ]
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        ## reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(bottom[0].shape)
        print('top 0 shape {} selfdata shape {}'.format(top[0].shape,bottom[0].shape))

    def next_idx(self):
        pass

    def forward(self, bottom, top):
        top[0].data = bottom[0].data
        data = top[0].data
        print('data shape:'+str(data.shape))
        firstvals = data[0,:,0,0]
        print('data first vals:'+str(firstvals))

    def backward(self, top, propagate_down, bottom):
        pass
