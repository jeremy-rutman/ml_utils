__author__ = 'jeremy'
import os
import logging
import cv2
import random
import socket
import traceback
import pymongo
import bson
import time
from trendi.constants import db
from trendi.constants import tamara_berg_categories

logging.basicConfig(level=logging.WARNING)

#as described here http://guanghan.info/blog/en/my-works/train-yolo/
#we want file w lines like:
#  box1_x1_ratio box1_y1_ratio box1_width_ratio box1_height_ratio
#these aparently correspond to filename lines in the train.txt file
# so we'll step thru dirs and append the train.txt file as well as the bbox file

#Note that each image corresponds to an annotation file. But we only need one single training list of images.
# Remember to put the folder 'images' and folder 'annotations' in the same parent directory,
# as the darknet code look for annotation files this way (by default).

def bbs_to_db(dir_of_bbfiles,dir_of_images,use_visual_output=True):
    '''  takes dir of bbs and images , puts info into db
    apparently mogno / pymongo creates the _id field so i dont have to
    :param dir_of_bbfiles:
    :param dir_of_images:
    :param use_visual_output:
    :return:
    '''
    imgfiles = [f for f in os.listdir(dir_of_images) if os.path.isfile(os.path.join(dir_of_images,f)) and f[-4:]=='.jpg' or f[-5:]=='.jpeg' ]
    for imgfile in imgfiles:
        corresponding_bbfile=imgfile.split('photo_')[1]
        corresponding_bbfile=corresponding_bbfile.split('.jpg')[0]
        corresponding_bbfile = corresponding_bbfile + '.txt'
        full_filename = os.path.join(dir_of_bbfiles,corresponding_bbfile)
        print('img {} bbfile {} full {}'.format(imgfile,corresponding_bbfile,full_filename))
        full_imgname = os.path.join(dir_of_images,imgfile)
        img_arr = cv2.imread(full_imgname)
        h,w = img_arr.shape[0:2]
#            master_bb.write(full_filename+' '+str(w)+' ' +str(h)+'\n')
        info_dict = {}
        info_dict['url'] = full_imgname
        info_dict['image_width'] = w
        info_dict['image_height'] = h
        items = []
        with open(full_filename,'r+') as fp:
            n_boxes = 0
            for line in fp:
                item_dict = {}
                cv2.imwrite
                n_boxes += 1
             #   line = str(category_number)+' '+str(  dark_bb[0])[0:n_digits]+' '+str(dark_bb[1])[0:n_digits]+' '+str(dark_bb[2])[0:n_digits]+' '+str(dark_bb[3])[0:n_digits] + '\n'
                vals = [int(s) if s.isdigit() else float(s) for s in line.split()]
                classno = vals[0]
                item_dict['category'] = tamara_berg_categories[classno]
                bb = [vals[1],vals[2],vals[3],vals[4]]
                print('classno {} ({}) bb {} imfile {} n_boxes {}'.format(classno,item_dict['category'],bb,imgfile,n_boxes))
#                bb = convert_dark_to_xywh((w,h),dark_bb)
                item_dict['bb'] = bb
        #         if use_visual_output:
        #             cv2.rectangle(img_arr, (bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),color=[int(255.0/10*classno),100,100],thickness=10)
        #             #resize to avoid giant images
        #             dest_height = 200
        #             dest_width = int(float(dest_height)/h*w)
        # #            print('h {} w{} destw {} desth {}'.format(h,w,dest_width,dest_height))
        #             factor = float(h)/dest_width
        #             newx = int(bb[0]*factor)
        #             newy = int(bb[0]*factor)
        #             im2 = cv2.resize(img_arr,(dest_width,dest_height))
        #             cv2.putText(im2,tamara_berg_categories[classno], (newx+1,newy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [int(255.0/10*classno),100,100],3)
        #
        #             cv2.imshow(imgfile,im2)
        #             cv2.waitKey(100)
                items.append(item_dict)
            # cv2.destroyAllWindows()
        #fp.close()
        info_dict['items'] = items
        print('db entry:'+str(info_dict))
        ack = db.training_images.insert_one(info_dict)
        print('ack:'+str(ack.acknowledged))

'''   db.training_images.find_one_and_update({'person_id': person_id},
                                              {'$set': {'gender': gender, 'status': 'done'}})
    image = db.genderator.find_one_and_update({'status': 'fresh'},
                                                  {'$set' : {'status': 'busy'}},
                                                  return_document=pymongo.ReturnDocument.AFTER)
'''


def bbs_to_txtfile(dir_of_bbfiles,dir_of_images,master_bbfile='/home/jeremy/dataset/master_bbfile.txt',use_visual_output=True):
    '''
    master bbfile format :
        [delete] path   imwidth, imheight
        category bbx bby bbwidth bbheight
        category bbx bby bbwidth bbheight
    :param dir_of_bbfiles:
    :param dir_of_images:
    :param master_bbfile:
    :return:
    '''
    imgfiles = [f for f in os.listdir(dir_of_images) if os.path.isfile(os.path.join(dir_of_images,f)) and f[-4:]=='.jpg' or f[-5:]=='.jpeg' ]
    with open(master_bbfile,'w') as master_bb:
        master_bb.write('[delete] path_to_file image_width image_height . . .\n')
        master_bb.write('category# x,y,w,h . .\n')
        for imgfile in imgfiles:
            corresponding_bbfile=imgfile.split('photo_')[1]
            corresponding_bbfile=corresponding_bbfile.split('.jpg')[0]
            corresponding_bbfile = corresponding_bbfile + '.txt'
            full_filename = os.path.join(dir_of_bbfiles,corresponding_bbfile)
            print('img {} bbfile {} full {}'.format(imgfile,corresponding_bbfile,full_filename))
            full_imgname = os.path.join(dir_of_images,imgfile)
            img_arr = cv2.imread(full_imgname)
            h,w = img_arr.shape[0:2]
            master_bb.write(full_filename+' '+str(w)+' ' +str(h)+'\n')
            with open(full_filename,'r+') as fp:
                n_boxes = 0
                for line in fp:
                    n_boxes += 1
                 #   line = str(category_number)+' '+str(  dark_bb[0])[0:n_digits]+' '+str(dark_bb[1])[0:n_digits]+' '+str(dark_bb[2])[0:n_digits]+' '+str(dark_bb[3])[0:n_digits] + '\n'
                    vals = [int(s) if s.isdigit() else float(s) for s in line.split()]
                    classno = vals[0]
                    dark_bb = [vals[1],vals[2],vals[3],vals[4]]
                    print('classno {} darkbb {} imfile {} n_boxes {}'.format(classno,dark_bb,imgfile,n_boxes))
                    bb = convert_dark_to_xywh((w,h),dark_bb)
                    master_bb.write(str(vals[0])+' '+str(bb[0])+' '+str(bb[1])+' '+str(bb[2])+' '+str(bb[3])+'\n')
                    if use_visual_output:
                        cv2.rectangle(img_arr, (bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),color=[int(255.0/10*classno),100,100],thickness=10)
                        #resize to avoid giant images
                        dest_height = 400
                        dest_width = int(float(dest_height)/h*w)
            #            print('h {} w{} destw {} desth {}'.format(h,w,dest_width,dest_height))
                        im2 = cv2.resize(img_arr,(dest_width,dest_height))
                        cv2.imshow(imgfile,im2)
                        cv2.waitKey(100)
                        cv2.destroyAllWindows()

def get_image_and_bbs(image_index_in_file,image_dir= '/home/jeremy/images',master_bbfile='/home/jeremy/master_bbfile.txt'):
    dir = '/home/jeremy/images'
    #resize to avoid giant images
    dest_height = 400
    dest_width = int(float(dest_height)/h*w)
    info = []
    i = 0
    with open(master_bbfile,'w') as master_bb:
        for line in master_bb:
            new_image=False
            line_elements = line.split()
            if len(line_elements) == 3:  #filename imwidth imheight
                new_image = True
                dict = {}
                dict['image_filename']=line_elements[0]
                dict['image_width']=line_elements[1]
                dict['image_height']=line_elements[2]
                logging.debug('new img {}'.format(dict))
            elif len(line_elements) == 4:  #delete filename width height
                delete = True
                new_image = True
                dict = {}
                dict['image_filename']=line_elements[1]
                dict['image_width']=line_elements[2]
                dict['image_height']=line_elements[3]
                logging.debug('new img TO DELETE {}'.format(dict))
            elif len(line_elements) == 5: #category x y w h
                new_image = False
                dict = {}
                dict['class']=line_elements[0]
                dark_bbox = [float(line_elements[1]),float(line_elements[2]),float(line_elements[3]),float(line_elements[4])]
                imsize = (dict['image_width'],dict['image_height'])
                bbox = convert_dark_to_xywh(imsize,dark_bbox)
                dict['bbox'] = bbox
                dict['image_height']=line_elements[3]
                new_image = False
                logging.debug('class {} bbox {}'.format(dict['class'],dict['bbox']))
            if i==image_index_in_file:
                return(dict)
            if new_image:
                i = i + 1
#            print('h {} w{} destw {} desth {}'.format(h,w,dest_width,dest_height))
#    im2 = cv2.resize(img_arr,(dest_width,dest_height))
#    cv2.imshow(imgfile,im2)
#   cv2.waitKey(0)

def do_delete(image_number):
    braini2_ip = '37.58.101.173'
    print('finished command')
    command = 'ssh root@'+braini2_ip+' mv /home/jeremy/trainjr.txt /home/jeremy/MOVEDITBABY.txt'
    os.system(command)
    print('finished command')
    with open('out.txt','a') as f:
        f.write('did '+command+'\n')
        f.close()

def show_regular_bbs(dir_of_bbfiles,dir_of_images):
    imgfiles = [f for f in os.listdir(dir_of_images) if os.path.isfile(os.path.join(dir_of_images,f)) and f[-4:]=='.jpg' or f[-5:]=='.jpeg' ]
    for imgfile in imgfiles:
        corresponding_bbfile=imgfile.split('photo_')[1]
        corresponding_bbfile=corresponding_bbfile.split('.jpg')[0]
        corresponding_bbfile = corresponding_bbfile + '.txt'
        full_filename = os.path.join(dir_of_bbfiles,corresponding_bbfile)
        print('img {} bbfile {} full {}'.format(imgfile,corresponding_bbfile,full_filename))
        full_imgname = os.path.join(dir_of_images,imgfile)
        img_arr = cv2.imread(full_imgname)
        if img_arr is None:
            logging.warning('coulndt open '+str(full_imgname))
            return
        h,w = img_arr.shape[0:2]

        with open(full_filename,'r+') as fp:
            n_boxes = 0
            for line in fp:
                n_boxes += 1
             #   line = str(category_number)+' '+str(  dark_bb[0])[0:n_digits]+' '+str(dark_bb[1])[0:n_digits]+' '+str(dark_bb[2])[0:n_digits]+' '+str(dark_bb[3])[0:n_digits] + '\n'
                vals = [int(s) if s.isdigit() else float(s) for s in line.split()]
                classno = vals[0]
                bb = [vals[1],vals[2],vals[3],vals[4]]
                print('classno {} ({}) darkbb {} imfile {} n_boxes {}'.format(classno,tamara_berg_categories[classno],bb,imgfile,n_boxes))
                cv2.rectangle(img_arr, (bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),color=[int(255.0/10*classno),100,100],thickness=10)
                cv2.putText(img_arr,tamara_berg_categories[classno], (bb[0]+1,bb[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 4, [int(255.0/10*classno),100,100],2)
            #resize to avoid giant images
            dest_height = 400
            dest_width = int(float(dest_height)/h*w)
#            print('h {} w{} destw {} desth {}'.format(h,w,dest_width,dest_height))
            im2 = cv2.resize(img_arr,(dest_width,dest_height))
            cv2.imshow(imgfile,im2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def show_darknet_bbs(dir_of_bbfiles,dir_of_images):
    imgfiles = [f for f in os.listdir(dir_of_images) if os.path.isfile(os.path.join(dir_of_images,f)) and f[-4:]=='.jpg' or f[-5:]=='.jpeg' ]
    for imgfile in imgfiles:
        corresponding_bbfile=imgfile.split('photo_')[1]
        corresponding_bbfile=corresponding_bbfile.split('.jpg')[0]
        corresponding_bbfile = corresponding_bbfile + '.txt'
        full_filename = os.path.join(dir_of_bbfiles,corresponding_bbfile)
        print('img {} bbfile {} full {}'.format(imgfile,corresponding_bbfile,full_filename))
        with open(full_filename,'r+') as fp:
            n_boxes = 0
            for line in fp:
                n_boxes += 1
             #   line = str(category_number)+' '+str(  dark_bb[0])[0:n_digits]+' '+str(dark_bb[1])[0:n_digits]+' '+str(dark_bb[2])[0:n_digits]+' '+str(dark_bb[3])[0:n_digits] + '\n'
                vals = [int(s) if s.isdigit() else float(s) for s in line.split()]
                classno = vals[0]
                dark_bb = [vals[1],vals[2],vals[3],vals[4]]
                print('classno {} darkbb {} imfile {} n_boxes {}'.format(classno,dark_bb,imgfile,n_boxes))
                full_imgname = os.path.join(dir_of_images,imgfile)
                img_arr = cv2.imread(full_imgname)
                h,w = img_arr.shape[0:2]
                bb = convert_dark_to_xywh((w,h),dark_bb)
                cv2.rectangle(img_arr, (bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),color=[int(255.0/10*classno),100,100],thickness=10)
            #resize to avoid giant images
            dest_height = 400
            dest_width = int(float(dest_height)/h*w)
#            print('h {} w{} destw {} desth {}'.format(h,w,dest_width,dest_height))
            im2 = cv2.resize(img_arr,(dest_width,dest_height))
            cv2.imshow(imgfile,im2)
            cv2.waitKey(0)

def dir_of_dirs_to_darknet(dir_of_dirs, trainfile,positive_filter=None,maxfiles_per_dir=999999,bbfile_prefix=None):
    initial_only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir)) ]
 #   print(str(len(initial_only_dirs))+' dirs:'+str(initial_only_dirs)+' in '+dir_of_dirs)
    # txn is a Transaction object
    #prepare directories
    initial_only_dirs.sort()
    only_dirs = []
    category_number = 0
    if dir_of_dirs[-1] == '/':
        dir_of_dirs = dir_of_dirs[0:-1]
    one_dir_up = os.path.split(dir_of_dirs)[0]
    print('outer dir of dirs:{} trainfile:{} '.format(dir_of_dirs,trainfile))
    print('dirs {}'.format(initial_only_dirs))
    for a_dir in initial_only_dirs:
        #only take 'test' or 'train' dirs, if test_or_train is specified
        print('doing directory {} class {} ({})'.format(a_dir,category_number,tamara_berg_categories[category_number]))
        raw_input('ret to cont')
        if (not positive_filter or positive_filter in a_dir):
            fulldir = os.path.join(dir_of_dirs,a_dir)
            only_dirs.append(fulldir)
            annotations_dir = os.path.join(one_dir_up,'labels')
            annotations_dir = os.path.join(annotations_dir,a_dir)
            ensure_dir(annotations_dir)
            n_files = dir_to_darknet(fulldir,trainfile,category_number,annotations_dir,maxfiles_per_dir=maxfiles_per_dir,bbfile_prefix=bbfile_prefix)
            print('did {} files in {}'.format(n_files,a_dir))
            category_number += 1
        else:
            print('directory filtered, didnt have {}'.format(positive_filter))

def dir_to_darknet(dir, trainfile,category_number,annotations_dir,randomize=True,maxfiles_per_dir=999999,bbfile_prefix=None):
    only_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    new_length = min(len(only_files),maxfiles_per_dir)
    only_files = only_files[0:new_length]
    if randomize:
        random.shuffle(only_files)
    n = 0
    n_digits = 5
    totfiles = len(only_files)
    with open(trainfile,'a') as fp_t:
        for a_file in only_files:

            containing_dir = os.path.split(dir)[-1]
            bbfilename_from_outerdir = False
#            if bbfilename_from_outerdir:
#                bbfile = containing_dir +"_{0:0>3}".format(n)+'.txt'  #dir_003.txt
#                bbfile = os.path.join(dir,bbfile)              #dir_of_dirs/dir/dir_003.txt
#                if bbfile_prefix:
#                    bbfile = bbfile_prefix+"_{0:0>3}".format(n)+'.txt'
#            else:
#                filebase = a_file[0:-4]   #file.jpg -> file
#                bbfile = filebase+"_{0:0>3}".format(n)+'.txt'
#                bbfile = os.path.join(dir,bbfile)              #dir_of_dirs/dir/dir_003.txt
            filebase = a_file[0:-4]   #file.jpg -> file
            bbfile = filebase+'.txt'
            bbfile = os.path.join(annotations_dir,bbfile)              #dir_of_dirs/dir/dir_003.txt
            print('bbfilename:{}, {} of {}'.format(bbfile,n,totfiles))
            with open(bbfile,'w') as fp_bb:
                full_filename = os.path.join(dir,a_file)
                dark_bb = get_darkbb(full_filename)
                if dark_bb is None:
                    continue
#                line = str(category_number)+join(str(a for a in dark_bb))
                line = str(category_number)+' '+str(dark_bb[0])[0:n_digits]+' '+str(dark_bb[1])[0:n_digits]+' '+str(dark_bb[2])[0:n_digits]+' '+str(dark_bb[3])[0:n_digits] + '\n'
                print('line to write:'+line)
                fp_bb.write(line)
                fp_t.write(full_filename+'\n')
                n = n + 1
#                raw_input('enter for next')
    #           fp_bb.flush()
                fp_bb.close()
        fp_t.close()
    return n


def dir_to_darknet_singlefile(dir, trainfile,bbfile,category_number,randomize=True,maxfiles_per_dir=999999):
    only_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    new_length = min(len(only_files),maxfiles_per_dir)
    only_files = only_files[0:new_length]
    if randomize:
        random.shuffle(only_files)
    n = 0
    n_digits = 5
    with open(trainfile,'a') as fp_t:
        with open(bbfile,'a') as fp_bb:
            for a_file in only_files:
                full_filename = os.path.join(dir,a_file)
                dark_bb = get_darkbb(full_filename)
                if dark_bb is None:
                    continue
#                line = str(category_number)+join(str(a for a in dark_bb))
                line = str(category_number)+' '+str(dark_bb[0])[0:n_digits]+' '+str(dark_bb[1])[0:n_digits]+' '+str(dark_bb[2])[0:n_digits]+' '+str(dark_bb[3])[0:n_digits] + '\n'
                print('line to write:'+line)
                fp_bb.write(line)
                fp_bb.close()
                fp_t.write(full_filename+'\n')
                n = n + 1
#                raw_input('enter for next')
    fp_t.close()
    return n


def get_darkbb(filename):
    base_filename = os.path.basename(filename)
    print('name:'+str(base_filename))
    if 'bbox_' in base_filename:
        strs = base_filename.split('bbox_')
        bb_str = strs[1]
        coords = bb_str.split('_')
        bb_x = int(coords[0])
        bb_y = int(coords[1])
        bb_w = int(coords[2])
        bb_h = coords[3].split('.')[0]  #this has .jpg or .bmp at the end
        bb_h = int(bb_h)
        bb=[bb_x,bb_y,bb_w,bb_h]
        logging.debug('bb:'+str(bb))
        if bb_h == 0:
            logging.warning('bad height encountered in imutils.resize_and_crop_image for '+str(filename))
            return None
        if bb_w == 0:
            logging.warning('bad width encountered in imutils.resize_and_crop_image for '+str(filename))
            return None
        img_arr = cv2.imread(filename)
        if img_arr is None:
            logging.warning('could not read {} , it reads as None'.format(filename))
            return None
        h,w = img_arr.shape[0:2]
        converted_bb = convert([w,h],bb)
        logging.debug('converted bb:{} im_w {} im_h {}'.format(converted_bb,w,h))
        w_conv = converted_bb[2]*w
        h_conv = converted_bb[3]*h
        x_conv = converted_bb[0]*w - w_conv/2
        y_conv = converted_bb[1]*h - h_conv/2
        logging.debug('converted bb xlates to: x:{} y:{} w:{} h:{}'.format(x_conv,y_conv,w_conv,h_conv))
        return(converted_bb)
    else:
        logging.warning('no bbox_ in filename, dont know how to get bb')


def convert(imsize, box):
    '''
    convert box [x y w h ] to box [x_center% ycenter% w% h%] where % is % of picture w or h
    :param size: [image_w,image_h]
    :param box: [x,y,w,h] of bounding box
    :return: [x_center% y_center% w% h%]
    '''
    dw = 1./imsize[0]
    dh = 1./imsize[1]
    x_middle = box[0] + box[2]/2.0
    y_middle = box[1] + box[3]/2.0
    w = box[2]
    h = box[3]
    logging.debug('x_mid {} y_mid {} w {} h {}'.format(x_middle,y_middle,w,h))
    x = x_middle*dw
    w = w*dw
    y = y_middle*dh
    h = h*dh
    return (x,y,w,h)

def convert_dark_to_xywh(imsize, dark_bb):
    '''
    convert box [x y w h ] to box [x_center% ycenter% w% h%] where % is % of picture w or h
    :param size: [image_w,image_h]
    :param box: [x,y,w,h] of bounding box
    :return: [x_center% y_center% w% h%]
    '''
    logging.debug('dark bb x_min {} y_mid {} w {} h {}  imw {} imh {}'.format(dark_bb[0],dark_bb[1],dark_bb[2],dark_bb[3],imsize[0],imsize[1]))

    w=imsize[0]
    h=imsize[1]
    x_middle = dark_bb[0]*w
    y_middle = dark_bb[1]*h
    bb_w = int(dark_bb[2]*w)
    bb_h = int(dark_bb[3]*h)
    x = int(x_middle - float(bb_w)/2)
    y = int(y_middle - float(bb_h)/2)
    bb = [x,y,bb_w,bb_h]
    logging.debug('output x {} y{} w {} h {} '.format(bb[0],bb[1],bb[2],bb[3]))
    return [x,y,bb_w,bb_h]

def ensure_dir(f):
    '''

    :param f: file or directory name
    :return: no return val, creates dir if it doesnt exist
    '''
    logging.debug('f:' + f)
    # d = os.path.dirname(f)
    if not os.path.exists(f):
        #        print('d:'+str(d))

        os.makedirs(f)


host = socket.gethostname()
print('host:'+str(host))

if __name__ == '__main__':
    if host == 'jr':
        dir_of_dirs = '/home/jeremy/python-packages/trendi/classifier_stuff/caffe_nns/dataset'
        images_dir = '/home/jeremy/tg/berg_test/cropped'
        dir = '/home/jeremy/tg/berg_test/cropped/test_pairs_belts'
        trainfile = '/home/jeremy/tg/trainjr.txt'
        bbfile = '/home/jeremy/tg/bbjr.txt'
        annotations_dir = '/home/jeremy/annotations'
    else:
        dir_of_dirs = '/home/jeremy/dataset/'
        images_dir = '/home/jeremy/dataset/images'
        bbfiles = '/home/jeremy/dataset/bbs'
        dir = '/home/jeremy/tg/berg_test/cropped/test_pairs_belts'
        trainfile =  '/home/jeremy/core/classifier_stuff/caffe_nns/trainfilejr.txt'
        annotations_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/annotations'

#    show_regular_bbs(bbfiles,images_dir)

    bbs_to_db(bbfiles,images_dir,use_visual_output=True)
 #   n_files = dir_of_dirs_to_darknet(images_dir,trainfile,maxfiles_per_dir=50000,positive_filter='train')
 #   n_files = dir_of_dirs_to_darknet(images_dir,trainfile,maxfiles_per_dir=50000,positive_filter='test')
#    n_files = dir_to_darknet(dir,trainfile,bbfile,37)
