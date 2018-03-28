import cv2
import numpy as np
# import scipy as sp
import os
import logging
import time
import string
import random
import copy

import imutils
import read_various_training_formats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_example")
logger.setLevel(logging.DEBUG)

def generate_images(img_filename, max_angle = 5,n_angles=10,
                    max_offset_x = 100,n_offsets_x=1,
                    max_offset_y = 100, n_offsets_y=1,
                    max_scale=1.2, n_scales=1,
                    noise_level=0.05,n_noises=1,noise_type='gauss',
                    max_blur=2, n_blurs=1,
                    do_mirror_lr=True,do_mirror_ud=False,output_dir=None,
                    show_visual_output=False,bb=None,do_bb=False,suffix='.jpg'):
    '''
    generates a bunch of variations of image by rotating, translating, noising etc
    total # images generated is n_angles*n_offsets_x*n_offsets_y*n_noises*n_scales*etc, these are done in nested loops
    if you don't want a particular xform set n_whatever = 0
    original image dimensions are preserved
    :param img_arr: image array to vary
    :param max_angle: rotation limit (degrees)
    :param n_angles: number of rotated images
    :param max_offset_x: x offset limit (pixels)
    :param n_offsets_x: number of x-offset images
    :param max_offset_y: y offset limit (pixels)
    :param n_offsets_y: number of y-offset images
    :param max_scales: global scaling factor
    :param n_scales: number of globally scaled images
    :param noise_level: level of gaussian noise to add - 0->no noise, 1->noise_level (avg 128)
    :param n_noises: number of noised images
    :param noise_type     'gauss'     Gaussian-distributed additive noise.
                                            'poisson'   Poisson-distributed noise generated from the data.
                                            's&p'       Replaces random pixels with 0 or 1.
                                            'speckle'   Multiplicative noise using out = image + n*image
                                            None
    :param max_blur: level of blur (pixels in kernel) to add - 0->no noise,
    :param n_blurs: number of blurred images
    :param do_mirror_lr: work on orig and x-axis-flipped copy
    :param do_mirror_ud: work on orig and x-axis-flipped copy
    :param output_dir: dir to write output images
    :return:
    '''

    img_arr = cv2.imread(img_filename)
    if img_arr is None:
        logging.warning('didnt get input image '+str(img_filename))
        return
    orig_path, filename = os.path.split(img_filename)
    if output_dir is not None and not os.path.exists(output_dir):
        os.mkdir(output_dir)
    eps = 0.01
    if n_angles <2:
        angles = [0.0]
    else:
        angles = np.arange(-max_angle, max_angle+eps, max_angle*2 / (n_angles-1))
    if n_offsets_x <2:
        offsets_x = [0]
    else:
        offsets_x = np.arange(-max_offset_x, max_offset_x+eps, max_offset_x*2/(n_offsets_x-1))
    if n_offsets_y <2:
        offsets_y = [0]
    else:
        offsets_y = np.arange(-max_offset_y, max_offset_y+eps, max_offset_y*2/(n_offsets_y-1))
    if n_scales <1:
        scales = [1.0]
    elif n_scales ==1:  #todo - change dx , dy , angles to have ==1 case
        scales = [max_scale]
    else:
        scales = np.arange(1, max_scale+eps, (max_scale-1)/(n_scales-1))
    if n_blurs <1:
        blurs = [0]
    elif n_blurs ==1:
        blurs = [max_blur]
    else:
        print('n_blurs-1:' + str(n_blurs-1))
        rat = float(max_blur)/(n_blurs-1)
        print('rat:'+str(rat))
        blurs = np.arange(1, max_blur+eps, rat)
    if n_noises <1:
         n_noises=1
         noise_type=None
    print('angles {0} offsets_x {1} offsets_y {2} scales {3} n_noises {4} lr {5} ud {6} blurs {7} '.format(angles,offsets_x,offsets_y,scales,n_noises,do_mirror_lr,do_mirror_ud,blurs))

    height=img_arr.shape[0]
    width=img_arr.shape[1]
    if len(img_arr.shape) == 2:
        depth = img_arr.shape[2]
    else:
        depth = 1
    center = (width/2,height/2)
    reflections=[img_arr]
    if do_mirror_lr:
        fimg=img_arr.copy()
        mirror_image = cv2.flip(fimg,1)
        reflections.append(mirror_image)
    if do_mirror_ud:
        fimg=img_arr.copy()
        mirror_image = cv2.flip(fimg,0)
        reflections.append(mirror_image)
    if do_mirror_ud and do_mirror_lr:
        fimg=img_arr.copy()
        mirror_image = cv2.flip(fimg,0)
        mirror_image = cv2.flip(mirror_image,1)
        reflections.append(mirror_image)
    if show_visual_output:
        cv2.imshow('orig',img_arr)
        k = cv2.waitKey(0)
    if 'bbox_' in img_filename and bb is None and do_bb:
        strs = img_filename.split('bbox_')
        bb_str = strs[1]
        coords = bb_str.split('_')
        bb_x = int(coords[0])
        bb_y = int(coords[1])
        bb_w = int(coords[2])
        bb_h = coords[3].split('.')[0]  #this has .jpg or .bmp at the end
        bb_h = int(bb_h)
        bb=[bb_x,bb_y,bb_w,bb_h]
        bb_points  = [[bb_x,bb_y],[bb_x+bb_w,bb_y],[bb_x,bb_y+bb_h],[bb_x+bb_w,bb_y+bb_h]]  #topleft topright bottomleft bottomright
        print('bb:'+str(bb))
        if bb_h == 0:
            logging.warning('bad height encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
            return None
        if bb_w == 0:
            logging.warning('bad width encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
            return None

# Python: cv2.transform(src, m[, dst]) -> dst
#http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#void%20transform%28InputArray%20src,%20OutputArray%20dst,%20InputArray%20m%29


    #SO CLEANNNN
    for n_reflection in range(0,len(reflections)):
        for offset_x in offsets_x:
            for offset_y in offsets_y:
                for angle in angles:
                    for scale in scales:
                        for i in range(0,n_noises):
                            for blur in blurs:
                                original_img = reflections[n_reflection]
                                if blur == 0:
                                    blurred = original_img  #blur=0 crashes cv2.blur
                                else:
                                    blurred = cv2.blur(original_img,(int(blur),int(blur)))   #fails if blur is nonint or 0
                                noised = add_noise(blurred,noise_type,noise_level)
                                print('center {0} angle {1} scale {2} h {3} w {4}'.format(center,angle, scale,height,width))
                                M = cv2.getRotationMatrix2D(center, angle,scale)
#                                print('M='+str(M))
                                M[0,2]=M[0,2]+offset_x
                                M[1,2]=M[1,2]+offset_y
                                print('M='+str(M))
                                dest = np.ones_like(img_arr) * 255
#                                xformed_img_arr  = cv2.warpAffine(noised,  M, (width,height),dst=dest,borderMode=cv2.BORDER_TRANSPARENT)
                                xformed_img_arr  = cv2.warpAffine(noised,  M, (width,height),dst=dest,borderMode=cv2.BORDER_REPLICATE)
                                xformed_img_arr = dest
                                if do_bb:
                                    xformed_bb_points  = np.dot(bb_points,M)
                                name = filename[0:-4]+'_ref{0}dx{1}dy{2}rot{3}scl{4}n{5}b{6}'.format(n_reflection,offset_x,offset_y,angle,scale,i,blur)+suffix
                                name = filename[0:-4]+'_m%dx%dy%dr%.2fs%.2fn%db%.2f' % (n_reflection,offset_x,offset_y,angle,scale,i,blur)+suffix
                                if output_dir is not None:
                                    full_name = os.path.join(output_dir,name)
                                else:
                                    full_name = os.path.join(orig_path,name)
                                print('name:'+str(full_name))
                                cv2.imwrite(full_name, xformed_img_arr)
                                if show_visual_output:
                                    cv2.imshow('xformed',xformed_img_arr)
                                    k = cv2.waitKey(0)

def multichannel_to_mask(multichannel_arr):
    '''
    from n-channel binary image (one chan for every category) make mask_array (single chan with integers indicating categories), make
    :param multichannel_arr:
    :return:
    '''
    if len(multichannel_arr.shape) != 3:
        logging.debug('got 1-chan image in multichannel_to_mask')
        return multichannel_arr
    h,w,c = multichannel_arr.shape
    output_arr = np.zeros([h,w])
    cumulative = 0
    for chan in range(c):
        nth_chan = multichannel_arr[:,:,chan]
        pixel_count = np.count_nonzero(nth_chan)
        cumulative = cumulative + pixel_count
#        print('multichannel to mask {} pixcount {}'.format(chan,pixel_count))
        output_arr[nth_chan != 0] = chan
        pixel_count = np.count_nonzero(output_arr)
#        print('cumulative pixcount {}'.format(cumulative))
    return output_arr

def mask_to_multichannel(mask_arr,n_channels):
    '''
    from mask_array (single chan with integers indicating categories), make n-channel binary image (one chan for every category)
    :param mask_arr:
    :param n_channels:
    :return:
    '''
    if len(mask_arr.shape) != 2:
        logging.debug('got multichannel image in mask_to_multichannel, converting to single chan: array shape:'+str(mask_arr.shape))
#        assert(mask_arr[:,:,0] == mask_arr[:,:,1])   #include these if paranoid
#        assert(mask_arr[:,:,0] == mask_arr[:,:,2])
        mask_arr = mask_arr[:,:,0]  #take 0th channel
    h,w = mask_arr.shape[0:2]
    output_arr = np.zeros([h,w,n_channels])

    for i in np.unique(mask_arr):
        channel = np.zeros([h,w])
        channel[mask_arr == i] = 1
 #       print('mask to multichannel {} pixcount {}'.format(i,pixel_count))
        output_arr[:,:,i] = channel
   #     print('cumulative pixcount {}'.format(pixel_count))
        logging.debug('nonzero elements in layer {}:{} '.format(i,len(mask_arr[mask_arr==i])))
        logging.debug('nonzero in multichan layer {}:{}'.format(i,np.count_nonzero(output_arr[:,:,i])))
    logging.debug('nonzero elements in orig:{} nonzero in multichan {}'.format(np.nonzero(mask_arr),np.nonzero(output_arr)))
    return output_arr
#

def resize_bbs(bblist_xywh,orig_shape,new_shape,img_arr=None):
    x_factor = new_shape[1]/orig_shape[1]
    y_factor = new_shape[0]/orig_shape[0]
    print('resize factor : {},{}'.format(x_factor,y_factor))
    resized_bbs = []
    for bb in bblist_xywh:
        bb_out = [bb[0]*x_factor,bb[1]*y_factor,bb[2]*x_factor,bb[3]*y_factor]
        resized_bbs.append(bb_out)
        if img_arr is not None:
            imutils.bb_with_text(img_arr, bb_out, 'resized')

def flip_bbs(image_dims_h_w, bb_list_xywh,flip_rl=False,flip_ud=False):
    for bb in bb_list_xywh:
        logging.debug('initial bb {}'.format(bb))
        if flip_rl:
            right_margin = image_dims_h_w[1]-(bb[0]+bb[2])    #width - right bb edge
            bb[0] = right_margin
        if flip_ud:
            bottom_margin = image_dims_h_w[0]-(bb[1]+bb[3])    #height - bottom bb edge
            bb[1] = bottom_margin
        logging.debug('final bb {}'.format(bb))
    return bb_list_xywh

def warp_bbs(bblist_xywh,M,dims_hw,img_arr=None):
    '''
    apply affine xfrom matrix m to bbs
    :param bblist_xywh:
    :param M:
    :return: bblist_xywh after affine xform
    '''
    bbs_out=[]
  #  bblist_xywh = [bblist_xywh[0]]
    logging.debug('Mshape '+str(M.shape))
    logging.debug('img dims '+str(dims_hw))
    for bb in bblist_xywh:
  #      bbs_xy_chans = np.array([[bb[0],bb[1]],[bb[0]+bb[2],bb[1]+bb[3]]])
  #      print('bbs out '+str(bbs_out))
#        print('cols {}'.format(M.cols))
#         src = np.array([
#             [bb[0],bb[1]],  #tl
#             [bb[0]+bb[2],bb[1]], #tr
#             [bb[0],bb[1]+bb[3]],  #bl
#             [bb[0]+bb[2],bb[1]+bb[3]]],  #br
#              dtype = "float32")
#      #   src = src.transpose()
#         print('sshape '+str(src.shape))
#         dst = np.dot(src,M[:,0:2]) +M[:,2]
#         print('dest from npdot:{}'.format(dst))
        src = np.array([
            [[bb[0],bb[1]]],  #tl
            [[bb[0]+bb[2],bb[1]]], #tr
            [[bb[0],bb[1]+bb[3]]],  #bl
            [[bb[0]+bb[2],bb[1]+bb[3]]]],  #br
             dtype = "float32")
        dst2=cv2.transform(src,M)
#        print('dst from cv2'+str(dst2))
        dst_bb = [(int(i[0][0]),int(i[0][1])) for i in dst2]
#        print('original dstbb '+str(dst_bb))
        minx=dst_bb[0][0]
        miny=dst_bb[0][1]
        maxx=dst_bb[0][0]
        maxy=dst_bb[0][1]
        for pt in dst_bb:
            if img_arr is not None:
                cv2.circle(img_arr,pt,10,(200,155,100))
            if pt[0]<minx:
                minx=pt[0]
            elif pt[0]>maxx:
                maxx=pt[0]
            if pt[1]<miny:
                miny=pt[1]
            elif pt[1]>maxy:
                maxy=pt[1]
        minx = max(0,minx)
        miny = max(0,miny)
        maxx = min(dims_hw[1],maxx)
        maxy = min(dims_hw[0],maxy)
        dst_bb=[minx,miny,maxx-minx,maxy-miny]
 #       print('dst_bb:'+str(dst_bb))
        bbs_out.append(dst_bb)
    # if img_arr is not None:
    #     cv2.imshow('circs',img_arr)
    #     cv2.waitKey(0)
  #  print('bbs out')
    return bbs_out


def test_warp_bbs(annotation_file='/home/jeremy/projects/core/images/female1_yololabels.txt',
                   img_file='/home/jeremy/projects/core/images/female1.jpg'):
    bbs,img_arr = read_various_training_formats.inspect_yolo_annotation(annotation_file,img_file)
    orig_img = copy.copy(img_arr)
    if img_arr is None:
        print('none img arr')
        return
    center = (img_arr.shape[0]/2,img_arr.shape[1]/2)
    angle = 20
    scale = 1.3
    offset_x = 20
    offset_y = 30
    M = cv2.getRotationMatrix2D(center, angle,scale)
#    logging.debug('db G')
    M[0,2]=M[0,2]+offset_x
    M[1,2]=M[1,2]+offset_y
    print('M:'+str(M))
    height,width=img_arr.shape[0:2]
    warped_image = cv2.warpAffine(img_arr,M,(width,height))
    warped_bbs = warp_bbs(bbs,M,img_arr.shape[0:2],img_arr=warped_image)
    for bb in warped_bbs:
        print('bb0 '+str(bb))
        warped_image = imutils.bb_with_text(warped_image, bb, 'out')
        # for pt in bb:
        #     cv2.circle(warped_image,pt,10,(100,255,100))
#     cv2.circle(warped_image,(bb[0,:]),10,(100,255,100))
#     cv2.circle(warped_image,(bb[1,:]),10,(100,100,255))
#     cv2.circle(warped_image,(bb[2,:]),10,(255,100,100))
#     cv2.circle(warped_image,(bb[3,:]),10,(255,255,100))
# #read_various_training_formats.show_annotations_xywh(warped_bbs,warped_image)
    cv2.imshow('out',warped_image)
    cv2.waitKey(0)


def test_flip_bbs(imgfile='images/female1.jpg'):
    img_arr=cv2.imread(imgfile)
    if img_arr is None:
        print('trouble getting '+imgfile)
        return
    h,w=img_arr.shape[0:2]
    bb1=[10,20,50,70]
    bb2=[100,200,100,200]
    bblist =[bb1,bb2]
    im2=cv2.flip(img_arr,1) #lr
    im2=cv2.flip(im2,0) #ud
    for bb in bblist:
        cv2.rectangle(img_arr,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),[100,200,255],thickness=1)
    cv2.imshow('orig',img_arr)
    cv2.waitKey(0)
    new_bbs = flip_bbs((h,w),bblist,flip_rl=True,flip_ud=True)
    for bb in new_bbs:
        cv2.rectangle(im2,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),[200,100,55],thickness=2)
    cv2.imshow('flip',im2)
    cv2.waitKey(0)

def crop_bblist(bblist_xywh,(height,width),(top,bottom,left,right)):
    new_bblist = []
    for bb in bblist_xywh:
        x1=bb[0]
        y1=bb[1]
        x2=bb[0]+bb[2]
        y2=bb[1]+bb[3]
        new_x1=x1-left if x1>left else 0
        new_y1=y1-top if y1>top else 0
        new_x2=x2-left if right>x2-left else right
        new_y2=y2-top if bottom>y2-top else bottom
        new_w = new_x2-new_x1 if  new_x1+ new_x2-new_x1 <= right-left else right-left-new_x1
        new_h = new_y2-new_y1 if  new_y1+new_y2-new_y1 <= bottom-top else bottom-top - new_y1
        new_bb=[new_x1,new_y1,new_w,new_h]
        new_bblist.append(new_bb)
    return new_bblist

def test_crop_bblist(annotation_file='/home/jeremy/projects/core/images/female1_yololabels.txt',
                   img_file='/home/jeremy/projects/core/images/female1.jpg'):
    bbs,img_arr = read_various_training_formats.inspect_yolo_annotation(annotation_file,img_file)
    orig_img = copy.copy(img_arr)
    if img_arr is None:
        print('none img arr')
        return
    height,width=img_arr.shape[0:2]
    top = 160
    bottom = height - 360
    left = 290
    right = width -270
    cropped_image = img_arr[top:bottom,left:right]
    print('cropped size '+str(cropped_image.shape))
    cropped_bbs = crop_bblist(bbs,(height,width),(top,bottom,left,right))
    for bb in cropped_bbs:
        print('bb0 '+str(bb))
        assert(bb[0]>=0),'x1 < 0'
        assert(bb[1]>=0),'y1 < 0'
        assert(bb[2]+bb[0]<=cropped_image.shape[1]),'x2 > w'
        assert(bb[3]+bb[1]<=cropped_image.shape[0]),'y2 > h'
        cropped_image = imutils.bb_with_text(cropped_image, bb, 'cropped', boxcolor=[255, 255.200])
        # for pt in bb:
        #     cv2.circle(warped_image,pt,10,(100,255,100))
#     cv2.circle(warped_image,(bb[0,:]),10,(100,255,100))
#     cv2.circle(warped_image,(bb[1,:]),10,(100,100,255))
#     cv2.circle(warped_image,(bb[2,:]),10,(255,100,100))
#     cv2.circle(warped_image,(bb[3,:]),10,(255,255,100))
# #read_various_training_formats.show_annotations_xywh(warped_bbs,warped_image)
    cv2.imshow('out',cropped_image)
    cv2.waitKey(0)


def generate_image_onthefly(img_filename_or_nparray, gaussian_or_uniform_distributions='uniform',
                   max_angle = 5,
                   max_offset_x = 5,max_offset_y = 5,
                   max_scale=1.2,min_scale=0.8,
                   max_noise_level= 0,noise_type='gauss',
                   max_blur=0,
                   max_color_rotation=0,
                   do_mirror_lr=True,do_mirror_ud=False,
                   crop_size=None,
                    show_visual_output=False,save_visual_output=False,mask_filename_or_nparray=None,n_mask_channels=21,
                            bblist_xywh=None):
    '''
    generates a bunch of variations of image by rotating, translating, noising etc
    total # images generated is n_angles*n_offsets_x*n_offsets_y*n_noises*n_scales*etc, these are done in nested loops
    if you don't want a particular xform set n_whatever = 0
    original image dimensions are preserved

    :param img_filename:
    :param gaussian_or_uniform_distributions:
    :param max_angle:
    :param max_offset_x:
    :param max_offset_y:
    :param max_scale: this is percent to enlarge/shrink image
    :param max_noise_level:
    :param noise_type:
    :param max_blur:
    :param do_mirror_lr:
    :param do_mirror_ud:
    :param output_dir:
    :param show_visual_output:
    :param suffix:
    :return:
    TODO
    add color shifting
    fix blur / noise
    ''' #
    start_time = time.time()
    if isinstance(img_filename_or_nparray,basestring):
 #       logging.debug('db A filename:'+img_filename_or_nparray)
        img_arr = cv2.imread(img_filename_or_nparray)
    else:
        img_arr = img_filename_or_nparray
    if img_arr is None:
        logging.warning('didnt get input image '+str(img_filename_or_nparray))
        return
    mask_arr = None
    if mask_filename_or_nparray is not None:
        if isinstance(mask_filename_or_nparray,basestring):
#            logging.debug('db A1 filename:'+mask_filename_or_nparray)
            mask_arr = cv2.imread(mask_filename_or_nparray)
        else:
            mask_arr = mask_filename_or_nparray
        if mask_arr is None:
            logging.warning('didnt get mask image '+str(mask_filename_or_nparray))
            return
#convert mask img to binary multichannel image
        mask_arr = mask_to_multichannel(mask_arr,n_mask_channels)

    #check that mask size and img size are equal
        if mask_arr.shape[0]!=img_arr.shape[0] or mask_arr.shape[1]!= img_arr.shape[1]:
            print('WARNING shape mismatch (no crop) in augment images, forcing reshape - imgshape {} maskshape {}'.format(img_arr.shape,mask_arr.shape))

    angle = 0
    offset_x = 0
    offset_y = 0
    scale = 0
    noise_level = 0
    blur = 0
    crop_dx = 0
    crop_dy = 0
    x_room = 0
    y_room = 0
    height,width = img_arr.shape[0:2]

    if crop_size:
        #WIP fix too small images here - embed into black bgnd of sufficient size
#        if img_arr.shape[0]<
        #calculate headroom left after crop. actual crop is random within that headroom iirc
        x_room = width - crop_size[1]
        y_room = height - crop_size[0]
        if x_room<0 or y_room<0:
            logging.debug('crop {} is larger than incoming image {} so I need to resize'.format(crop_size,img_arr.shape[0:2]))
            if x_room<y_room:
                factor = float(crop_size[1]+2)/width #assumes crop is x,y not y,x
                resize_size = (int(height*factor),crop_size[1])
            else:
                factor = float(crop_size[0]+2)/height #add 1 since rounding can cause output to be one pix too small
                resize_size = (crop_size[0],int(width*factor))
            logging.warning('resizing {} to {} so as to accomodate crop to {}'.format(img_arr.shape[0:2],resize_size,crop_size))
            img_arr= imutils.resize_keep_aspect(img_arr, output_size=resize_size, careful_with_the_labels=False) #img not labels
            if(mask_arr is not None):
  #              print('uniques beffg '+str(np.unique(mask_arr)))
                mask_arr= imutils.resize_keep_aspect(mask_arr, output_size=resize_size, careful_with_the_labels=True) #labels not img
   #             print('uniques aft '+str(np.unique(mask_arr)))

        height,width = img_arr.shape[0:2]
        x_room = width - crop_size[1]
        y_room = height - crop_size[0]
        if x_room<0 or y_room<0:
            logging.warning('crop {} is still larger than incoming image {} !!!!! something went wrong'.format(crop_size,img_arr.shape[0:2]))

 #       logging.debug('crop size {} xroom {} yroom {}'.format(crop_size,x_room,y_room))
#        if crop_size[0]!=img_arr.shape[0] or crop_size[1]!= img_arr.shape[1]:
##            print('WARNING shape mismatch with crop in augment images, forcing reshape!')
 #           print('img shape wxh {}x{} cropsize {}x{}'.format(img_arr.shape[0],img_arr.shape[1],crop_size[0],crop_size[1]))


    eps = 0.1

    if gaussian_or_uniform_distributions == 'gaussian':
        if max_angle:
            angle = np.random.normal(0,max_angle)
        if max_offset_x:
            offset_x = np.random.normal(0,max_offset_x)
        if max_offset_y:
            offset_y = np.random.normal(0,max_offset_y)
        if max_scale and min_scale:
            #         print('gscale limits {} {}'.format(1,np.abs(1.0-max_scale)/2))
            scale = max(eps,np.random.normal(max_scale+min_scale)/2.0,np.abs(max_scale-min_scale)/2.0) #make sure scale >= eps
        elif max_scale:
            #         print('gscale limits {} {}'.format(1,np.abs(1.0-max_scale)/2))
            scale = max(eps,np.random.normal(1,np.abs(1.0-max_scale)/2.0)) #make sure scale >= eps
        if max_noise_level:
            noise_level = max(0,np.random.normal(0,max_noise_level)) #noise >= 0
        if max_blur:
            blur = max(0,np.random.normal(0,max_blur)) #blur >= 0
        if x_room:
            crop_dx = max(-float(x_room)/2,int(np.random.normal(0,float(x_room)/2)))
            crop_dx = min(crop_dx,float(x_room)/2)
        if y_room:
            crop_dy = max(-float(y_room)/2,int(np.random.normal(0,float(y_room)/2)))
            crop_dy = min(crop_dy,float(y_room)/2)

    else:  #uniform distributed random numbers
        if max_offset_x:
            offset_x = np.random.uniform(-max_offset_x,max_offset_x)
        if max_offset_y:
            offset_y = np.random.uniform(-max_offset_y,max_offset_y)
        if max_scale and min_scale:
            #         print('gscale limits {} {}'.format(1,np.abs(1.0-max_scale)/2))
            scale = np.random.uniform(min_scale,max_scale) #make sure scale >= eps
        elif max_scale:
    #        print('uscale limits {} {}'.format(1-np.abs(1-max_scale),1+np.abs(1-max_scale)))
            scale = np.random.uniform(1-np.abs(1-max_scale),1+np.abs(1-max_scale))
        if max_noise_level:
            noise_level = np.random.uniform(0,max_noise_level)
        if max_blur:
            blur = np.random.uniform(0,max_blur)
        if x_room:
            crop_dx = int(np.random.uniform(0,float(x_room)/2))
        if y_room:
            crop_dy = int(np.random.uniform(0,float(y_room)/2))
        if max_angle:
            angle = np.random.uniform(-max_angle,max_angle)

    if len(img_arr.shape) == 3:
        depth = img_arr.shape[2]
    else:
        depth = 1
    center = (width/2,height/2)
  #  logging.debug('db C')

    flip_lr = 0
    flip_ud = 0
    if do_mirror_lr:
        flip_lr = np.random.randint(2)
    if do_mirror_ud:
        flip_ud = np.random.randint(2)
#    logging.debug('augment w {} h {} cropdx {} cropdy {} cropsize {} depth {} fliplr {} flipdud {} center {} angle {} scale {} offx {} offy {}'.format(
#        width,height,crop_dx,crop_dy,crop_size,depth,flip_lr,flip_ud,center,angle,scale,offset_x,offset_y))
    img_arr = do_xform(img_arr,width,height,crop_dx,crop_dy,crop_size,depth,flip_lr,flip_ud,blur,noise_level,center,angle,scale,offset_x,offset_y)
#    if show_visual_output:
#        logging.debug('img_arr shape:'+str(img_arr.shape))
#        cv2.imshow('xformed',img_arr)
#        k = cv2.waitKey(0)
    if mask_arr is not None:  #do xform to mask
      #  logging.debug('doing mask augmentation')
        mask_arr =do_xform(mask_arr,width,height,crop_dx,crop_dy,crop_size,depth,flip_lr,flip_ud,blur,noise_level,center,angle,scale,offset_x,offset_y)
        mask_arr = multichannel_to_mask(mask_arr)
        if save_visual_output:
            lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(30)]
            name = "".join(lst)
            cv2.imwrite(name+'.jpg',img_arr)
            maskname = name+'_mask.png'
            cv2.imwrite(maskname,mask_arr)
#        logging.debug('augment output:img arr size {} mask size {}'.format(img_arr.shape,mask_arr.shape))

    if bblist_xywh is not None:
        bblist_xywh = do_xform_bblist_xywh(bblist_xywh,width,height,crop_dx,crop_dy,crop_size,depth,flip_lr,flip_ud,blur,noise_level,center,angle,scale,offset_x,offset_y)


    if save_visual_output:
        lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(10)]
        name = "".join(lst)+'.jpg'
        cv2.imwrite(name,img_arr)
        if mask_arr is not None:
            cv2.imwrite(name.replace('.jpg','.png'),mask_arr)
    if show_visual_output:
        img_copy = copy.copy(img_arr)
        if mask_arr:
            labels = {i:str(i) for i in range(max(np.unique(mask_arr)))}
            imutils.show_mask_with_labels(mask_arr, labels, original_image=img_arr, visual_output=True)
        else:
            if bblist_xywh:
                for bb in bblist_xywh:
                    cv2.rectangle(img_copy,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),[255,50,100],thickness=2)
            cv2.imshow('augmented',img_copy)
            cv2.waitKey(0)

  #assuming that there is either mask or bblist not both
    if mask_arr is not None:
        return img_arr,mask_arr
    elif bblist_xywh is not None:
        return img_arr,bblist_xywh
    return img_arr


def do_xform(img_array,width,height,crop_dx,crop_dy,crop_size,depth,flip_lr,flip_ud,blur,noise_level,center,angle,scale,offset_x,offset_y):
    #todo this can all be cleaned up by putting more of the generate_image_on_thefly code here
#    logging.debug('db D')
    if flip_lr:
 #       logging.debug('db D1')
        img_array = cv2.flip(img_array,1)
 #       logging.debug('db D2')

    if flip_ud:
        img_array = cv2.flip(img_array,0)
#    logging.debug('db E')

# Python: cv2.transform(src, m[, dst]) -> dst
#http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#void%20transform%28InputArray%20src,%20OutputArray%20dst,%20InputArray%20m%29
    if blur:  #untested
        img_array = cv2.blur(img_array,(int(blur),int(blur)))   #fails if blur is nonint or 0

    if noise_level:  #untested
        img_array = add_noise(img_array,noise_type,noise_level)
#    logging.debug('db F')

  #  print('center {0} angle {1} scale {2} h {3} w {4} dx {5} dy {6} noise {7} blur {8}'.format(center,angle, scale,height,width,offset_x,offset_y,noise_level,blur))
    M = cv2.getRotationMatrix2D(center, angle,scale)
#    logging.debug('db G')
    M[0,2]=M[0,2]+offset_x
    M[1,2]=M[1,2]+offset_y
 #   print('M='+str(M))
#                                xformed_img_arr  = cv2.warpAffine(noised,  M, (width,height),dst=dest,borderMode=cv2.BORDER_TRANSPARENT)
    img_array  = cv2.warpAffine(img_array,  M, (width,height),borderMode=cv2.BORDER_REPLICATE)
    if crop_size:
        if crop_dx is None:
            crop_dx = 0
        if crop_dy is None:
            crop_dy = 0
        left = int(round(max(0,round(float(width-crop_size[1])/2) - crop_dx)))
        right = int(round(left + crop_size[1]))
        top = int(round(max(0,round(float(height-crop_size[0])/2) - crop_dy)))
        bottom = int(round(top + crop_size[0]))
        logging.debug('incoming wxh {}x{} cropsize {}'.format(width,height,crop_size))
 #       print('left {} right {} top {} bottom {} crop_dx {} crop_dy {} csize {} xroom {} yroom {}'.format(left,right,top,bottom,crop_dx,crop_dy,crop_size,x_room,y_room))
        if depth!=1:
            img_array = img_array[top:bottom,left:right,:]
            #print img_arr.shape
        else:
            img_array = img_array[top:bottom,left:right]
    return img_array
#  raw_input('enter to cont')

def do_xform_bblist_xywh(bb_list_xywh,width,height,crop_dx,crop_dy,crop_size,depth,flip_lr,flip_ud,blur,noise_level,center,angle,scale,offset_x,offset_y):
    #todo this can all be cleaned up by putting more of the generate_image_on_thefly code here
#    logging.debug('db D')
    logging.debug('augmenting bbs w {}  h {} dx {} dy {} crop {} lr {} ud {} center {} andgle {} scale {} offx {} offy {}'.
        format(width,height,crop_dx,crop_dy,crop_size,flip_lr,flip_ud,center,angle,scale,offset_x,offset_y))
    if flip_lr:
 #       logging.debug('db D1')
        flip_bbs((height,width),bb_list_xywh,flip_rl=True,flip_ud=False)
#        img_array = cv2.flip(img_array,1)
 #       logging.debug('db D2')

    if flip_ud:
        flip_bbs((height,width),bb_list_xywh,flip_rl=False,flip_ud=True)

    M = cv2.getRotationMatrix2D(center, angle,scale)
    M[0,2]=M[0,2]+offset_x
    M[1,2]=M[1,2]+offset_y

    bb_list_xywh = warp_bbs(bb_list_xywh,M,(height,width))
    if crop_size:
        if crop_dx is None:
            crop_dx = 0
        if crop_dy is None:
            crop_dy = 0
        left = int(round(max(0,round(float(width-crop_size[1])/2) - crop_dx)))
        right = int(round(left + crop_size[1]))
        top = int(round(max(0,round(float(height-crop_size[0])/2) - crop_dy)))
        bottom = int(round(top + crop_size[0]))
        logging.debug('incoming wxh {}x{} cropsize {}'.format(width,height,crop_size))
 #       print('left {} right {} top {} bottom {} crop_dx {} crop_dy {} csize {} xroom {} yroom {}'.format(left,right,top,bottom,crop_dx,crop_dy,crop_size,x_room,y_room))
        bb_list_xywh = crop_bblist(bb_list_xywh,(height,width),(top,bottom,left,right))
    return bb_list_xywh

def generate_images_for_directory(fulldir,**args):
    only_files = [f for f in os.listdir(fulldir) if os.path.isfile(os.path.join(fulldir, f))]
    for a_file in only_files:
        full_filename = os.path.join(fulldir,a_file)
        generate_images(full_filename,**args)

def generate_masks(img_filename, **kwargs):

    img_arr = cv2.imread(img_filename,cv2.IMREAD_GRAYSCALE)
    if img_arr is None:
        logging.warning('didnt get input image '+str(img_filename))
        return
    print('shape:'+str(img_arr.shape))
    if len(img_arr.shape) == 3:
        logging.warning('got 3 channel image '+str(img_filename)+', using first chan')
        img_arr = img_arr[:,:,0]
    if img_arr is None:
        logging.warning('didnt get input image '+str(img_filename))
        return
    h,w = img_arr.shape[0:2]
    uniques = np.unique(img_arr)
    n_uniques=len(uniques)
    binary_masks = np.zeros([h,w,n_uniques])
    for i in range(0,n_uniques):
        binary_masks[:,:,i] = img_arr[:,:]==uniques[i]
        cv2.imshow('mask'+str(i),binary_masks[:,:,i])
        transformed_mask = transform_image(binary_masks[:,:,i],kwargs)

    cv2.waitKey(0)

def generate_images_for_directory_of_directories(dir_of_dirs,filter= None,**args):
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))  ]
    logging.debug(str(only_dirs))
    if filter:
        only_dirs = [dir for dir in only_dirs if filter in dir  ]
    logging.debug(str(only_dirs))
    for a_dir in only_dirs:
        full_dir = os.path.join(dir_of_dirs,a_dir)
        generate_images_for_directory(full_dir,**args)

def clear_underutilized_bins(img_arr):
    h = np.histogram(img_arr,bins=57)
    print h

def add_noise(image, noise_typ,level):
    '''
    Parameters
    ----------
    image : ndarray
    Input image data. Will be converted to float.
    mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    '''
    print('adding noise type {0} level {1}'.format(noise_typ,level))
    if noise_typ == None:
        return image
    if noise_typ == "gauss":
        row,col,ch= image.shape
        print('row {} col {} ch {}'.format(row,col,ch))
        mean = 0
        var = level*255
        sigma = var**0.5
        print('sigma {0} mean {1}'.format(sigma,mean))
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        #        z=np.multiply(gauss,0)
 #       gauss = np.maximum(gauss,z)
        gauss = gauss.reshape(row,col,ch)
        #       cv2.imshow('orig',gauss)
  #     k = cv2.waitKey(0)
        noisy = (image + gauss)
        noisy =  noisy.astype(dtype=np.uint8)

        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = level
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = [255,255,255]
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = [0,0,0]
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

def augment_yolo_bbs(file_list='/media/jeremy/9FBD-1B00/data/jeremy/image_dbs/hls/voc_rio_udacity_test.txt',
        visual_output=False,replace_this=None,with_this=None,labels_dir=None,n_augmentations=3,path_filter=None,path_antifilter='rio'):
    '''
    takes a bunch of yolos and augments using generate_image_onthefly, right now for generating smaller objects
    :param file_list:
    :param visual_output:
    :param replace_this:
    :param with_this:
    :param labels_dir:
    :param n_augmentations:
    :param path_filter:   require filepaths to contain this string (to only augment certain dbs)
    :return:
    '''
    max_angle=7
    max_scale=1.1
    min_scale=0.7
    with open(file_list,'r') as fp:
        lines = fp.readlines()
        fp.close()
    print('{} lines in {}'.format(len(lines),file_list))
    for line in lines:
        if replace_this is not None:
            line = line.replace(replace_this,with_this)
        line = line.replace('\n','')
        print('got line '+str(line))
        if path_filter and not path_filter in line:
            logging.debug('didnt find {} in {}, skipping'.format(path_filter,line))
            continue
        if path_antifilter and path_antifilter in line:
            logging.debug('found {} in {}, skipping'.format(path_antifilter,line))
            continue
        tgdict = read_various_training_formats.yolo_to_tgdict(img_file=line,visual_output=visual_output,classlabels=constants.hls_yolo_categories,labels_dir_suffix=labels_dir)

        if tgdict is None:
            logging.warning('couldnt get dict for  {}, continuing to next'.format(line))
            continue

        annotations = tgdict['annotations']
        filename = tgdict['filename']
        logging.debug('file {}\nannotations {}'.format(filename,annotations))
        bbox_list = []
        if annotations == []:
            logging.info('no annotations, skipping')
            continue
        for annotation in annotations:
            bbox_xywh=annotation['bbox_xywh']
            bbox_list.append(bbox_xywh)
        for n in range(n_augmentations):
            logging.debug('\norig bbox list:'+str(bbox_list))
            bbox_to_send=copy.deepcopy(bbox_list) #func can modify arg it seems
            img_arr,new_bbox_list = generate_image_onthefly(filename,show_visual_output=visual_output,bblist_xywh=bbox_to_send,max_angle=max_angle,max_scale=max_scale,min_scale=min_scale)
            logging.debug('new bbox list:'+str(new_bbox_list))
            if img_arr is None:
                logging.warning('couldnt get {}, continuing to next'.format(filename))

            #write image
            if line[-5:] == '.jpeg':
                suffix = line[-5:]
            else:
                suffix = line[-4:]  #keep augmented and orig images of same type, people
            new_imgfile=line.strip(suffix)+'_aug'+str(n)+suffix
            new_lblfile=line.strip(suffix)+'_aug'+str(n)+'.txt'
            print('saving new image {}'.format(new_imgfile))
            cv2.imwrite(new_imgfile,img_arr)

            #write annotation
            tgdict['filename']=new_imgfile
            for i in range(len(annotations)):
                annotations[i]['bbox_xywh']=new_bbox_list[i]
            logging.debug('tgdict {}'.format(tgdict))
            read_various_training_formats.tgdict_to_yolo(tgdict)

            #write file list
            with open(file_list,'a') as fp2:
                fp2.write(new_imgfile+'\n')
                fp2.close()
     #   raw_input('ret to cont')

if __name__=="__main__":
    print('running main')
    img_filename = '../images/female1.jpg'
    img_filename = '../images/female1_256x256.jpg'
    image_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/train_200x150'
    label_dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_200x150'

    img = '/media/jeremy/9FBD-1B00/data/olympics/'

    file_list = '/data/jeremy/image_dbs/hls/insecam/07.05.2015_cameras_01-73filelist.txt'
    # augment_yolo_bbs(file_list='/data/jeremy/image_dbs/hls/kitti/training/yolo_train_test.txt',visual_output=True,
    #                  replace_this='/mnt/',with_this='/data/jeremy/image_dbs/')
    augment_yolo_bbs(file_list=file_list,visual_output=False)
#    test_crop_bblist()
#    test_warp_bbs()

    if(0):
        dir = '/home/jeremy/Dropbox/tg/pixlabels/test_256x256_novariations'
        images = [f for f in os.listdir(dir)]
        for image in images:
            label = image[:-4]+'.png'
            print('image {} label {}'.format(image,label))
            labelfile = os.path.join('/home/jeremy/Dropbox/tg/pixlabels/labels_256x256_novariations',label)
            imfile = os.path.join(dir,image)
            if os.path.isfile(imfile) and os.path.isfile(labelfile):
                in1 = cv2.imread(imfile)
                in2 = cv2.imread(labelfile)
                for i in range(10):
                    out1,out2 = generate_image_onthefly(in1, mask_filename_or_nparray=in2)
                    cv2.imwrite('out1.jpg',out1)
                    cv2.imwrite('out2.png',out2)
                    imutils.show_mask_with_labels('out2.png', labels=constants.ultimate_21, original_image='out1.jpg', visual_output=True)

    if(0):
        in1 = np.zeros([500,500,3])
        in2 = np.zeros_like(in1,dtype=np.uint8)
        for i in range(1,21):
            color = (np.random.randint(256),np.random.randint(256),np.random.randint(256))
            position = (50+np.random.randint(400),50+np.random.randint(400))
            radius = np.random.randint(200)
            cv2.circle(in1,position,radius,color=color,thickness=10)
            cv2.circle(in2,position,radius,(i,i,i),thickness=10)
            pt1 = (50+np.random.randint(400),50+np.random.randint(400))
            pt2 = (pt1[0]+np.random.randint(100),pt1[1]+np.random.randint(100))
            cv2.rectangle(in1,pt1,pt2,color,thickness=10)
            cv2.rectangle(in2,pt1,pt2,color=(i,i,i),thickness=10)
        cv2.imwrite('in1.jpg',in1)
        cv2.imwrite('in2.png',in2)
        imutils.show_mask_with_labels('in2.png', labels=constants.ultimate_21, visual_output=True, original_image='in1.jpg')
        cv2.destroyAllWindows()

        while(1):
    #        in2 = cv2.imread('/home/jeremy/Pictures/Berlin_Naturkundemuseum_Dino_Schaedel_posterized.png')
            out1,out2 = generate_image_onthefly(in1, mask_filename_or_nparray=in2)
            cv2.imwrite('out1.jpg',out1)
            cv2.imwrite('out2.png',out2)
            imutils.show_mask_with_labels('out2.png', labels=constants.ultimate_21, original_image='out1.jpg', visual_output=True)
            print('orig uniques {} nonzero {} mask uniques {} nonzero {} '.format(np.unique(out1),np.count_nonzero(out1),np.unique(out2),np.count_nonzero(out2)))
            print('')
            print('')
            cv2.destroyAllWindows()

    while(0):
        generate_image_onthefly(img_filename, gaussian_or_uniform_distributions='uniform',
                   max_angle = 5,
                    max_offset_x = 10,max_offset_y = 10,
                   max_scale=1.2,
                   max_noise_level=0,noise_type='gauss',
                   max_blur=0,
                   do_mirror_lr=True,do_mirror_ud=False,
                   crop_size=(224,224),
                   show_visual_output=True)

    if 0:
        generate_images_for_directory(image_dir,
                    max_angle = 10,n_angles=2,
                    max_offset_x = 10,n_offsets_x=2,
                    max_offset_y = 10, n_offsets_y=2,
                    max_scale=1.3, n_scales=2,
                    noise_level=0.1,noise_type='gauss',n_noises=0,
                    max_blur=5, n_blurs=0,
                    do_mirror_lr=True,do_mirror_ud=False,do_bb=False,suffix='.jpg')

        generate_images_for_directory(label_dir,
                    max_angle = 10,n_angles=2,
                    max_offset_x = 10,n_offsets_x=2,
                    max_offset_y = 10, n_offsets_y=2,
                    max_scale=1.3, n_scales=2,
                    noise_level=0.1,noise_type='gauss',n_noises=0,
                    max_blur=5, n_blurs=0,
                    do_mirror_lr=True,do_mirror_ud=False,do_bb=False,suffix='.png')


#    generate_images(img_filename, max_angle = 3,n_angles=2,
#                    max_offset_x = 50,n_offsets_x=2,
#                   max_offset_y = 50, n_offsets_y=2,
#                   max_scale=1.2, n_scales=2,
#                   noise_level=0.1,noise_type='gauss',n_noises=2,
#                    max_blur=5, n_blurs=2,
#                    do_mirror_lr=True,do_mirror_ud=False,output_dir='snorb')