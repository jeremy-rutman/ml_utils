from __future__ import print_function

__author__ = 'jeremy'

import sys
import os
import cv2
import logging
import time
logging.basicConfig(level=logging.INFO)  #debug is actually lower than info: critical/error/warning/info/debug
import shutil
# So this file can be imported on servers where joblib is not installed
try:
    from joblib import Parallel,delayed
except:
    pass
import numpy as np
import multiprocessing
import copy
#from trendi import constants
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
import subprocess
import inspect
import string
import random

import constants
#import background_removal
#from trendi.paperdoll import neurodoll_falcon_client


#from trendi import Utils


######################
#bounding box specific
######################


def intersectionOverUnion(r1, r2):
    '''
    r1,r2 in form xywh
    :param r1:
    :param r2:
    :return:
    '''
    # print(r1, r2)

    # a if test else b

    intersectionx = int(max(r1[0], r2[0]))
    intersectiony = int(max(r1[1], r2[1]))
    intersectionw = int(min(r1[0] + r1[2], r2[0] + r2[2])) - int(intersectionx)
    if intersectionw < 0:
        intersectionw = 0
    intersectionh = int(min(r1[1] + r1[3], r2[1] + r2[3])) - int(intersectiony)
    if intersectionh < 0:
        intersectionh = 0
        # intersectionh -= intersectiony;
        # print('r1:' + str(r1) + ' r2:' + str(r2) + ' x,y,w,h:' + str(intersectionx) + ',' + str(intersectiony) + ',' + str(
        # intersectionw) + ',' + str(
        # intersectionh))
    totarea = r1[2] * r1[3] + r2[2] * r2[3]  # this includes overlap twice
    intersectionarea = intersectionw * intersectionh
    totarea = totarea - intersectionarea  # now totarea includes overlap only once
    iou = float(intersectionarea) / float(totarea)
    print('totarea,intarea,iou:' + str(totarea) + ',' + str(intersectionarea) + ',' + str(iou))
    return (iou)

def intersectionOverMinArea(r1,r2):
    '''
    r1,r2 in form xywh
    :param r1:
    :param r2:
    :return:
    '''

    intersectionx = int(max(r1[0], r2[0]))
    intersectiony = int(max(r1[1], r2[1]))
    intersectionw = int(min(r1[0] + r1[2], r2[0] + r2[2])) - int(intersectionx)
    if intersectionw < 0:
        intersectionw = 0
    intersectionh = int(min(r1[1] + r1[3], r2[1] + r2[3])) - int(intersectiony)
    if intersectionh < 0:
        intersectionh = 0
        # intersectionh -= intersectiony;
        # print('r1:' + str(r1) + ' r2:' + str(r2) + ' x,y,w,h:' + str(intersectionx) + ',' + str(intersectiony) + ',' + str(
        # intersectionw) + ',' + str(
        # intersectionh))
    min_area=min(r1[2]*r1[3],r2[2]*r2[3])
    intersectionarea = intersectionw * intersectionh
    frac = float(intersectionarea) / float(min_area)
    print('min_area,intarea,frac:' + str(min_area) + ',' + str(intersectionarea) + ',' + str(frac))
    return (frac)

def combine_bbs(bb1_xywh,bb2_xywh):
    minx=min(bb1_xywh[0],bb2_xywh[0])
    maxx=max(bb1_xywh[0]+bb1_xywh[2],bb2_xywh[0]+bb2_xywh[2])
    miny=min(bb1_xywh[1],bb2_xywh[1])
    maxy=min(bb1_xywh[1]+bb1_xywh[3],bb2_xywh[1]+bb2_xywh[3])
    w=maxx-minx
    h=maxy-miny
    return(minx,miny,w,h)


def get_person_bb_from_face(face, image_shape):
    x, y, w, h,  = face

    mid_face_x = x + w/2
    p_width = 3.5 * w
    p_height = 8 * h

    # person bb x1,x2,y1,y2
    p_x1 = int(round(max(0, mid_face_x - p_width/2)))
    p_x2 = int(round(min(image_shape[1], mid_face_x + p_width/2)))
    p_y1 = y
    p_y2 = int(round(min(image_shape[0], y + p_height)))
    return [p_x1, p_y1, p_x2, p_y2]


def fix_bb_x1y1x2y2(bb_x1y1x2y2):
    '''fix out-of-order bbs (x2y2x1y1) or right top ,left bottom'''
    if bb_x1y1x2y2[0]>bb_x1y1x2y2[2]: #swap x1y1 w. x2y2
        tmp=bb_x1y1x2y2[0]  #swap x1,x2
        bb_x1y1x2y2[0]=bb_x1y1x2y2[2]
        bb_x1y1x2y2[2]=tmp
        tmp=bb_x1y1x2y2[1]  #swap y1,y2
        bb_x1y1x2y2[1]=bb_x1y1x2y2[3]
        bb_x1y1x2y2[3]=tmp
    if bb_x1y1x2y2[1]>bb_x1y1x2y2[3]:  # bb is top right instead of top left or something
        logging.warning('malformed x1y1x2y2 bb {}, swapping y1 and y2'.format(bb_x1y1x2y2))
  #      raw_input('ret to cont')
        tmp=bb_x1y1x2y2[1]  #swap y1,y2, dont swap x
        bb_x1y1x2y2[1]=bb_x1y1x2y2[3]
        bb_x1y1x2y2[3]=tmp
#    print(bb_x1y1x2y2)
    return bb_x1y1x2y2



def bb_to_mask(bb, img_array):
    '''
    bb in form of x,y,w,h converted to np array the same size as img_array
    :param bb:
    :return:
    '''
    h, w = img_array.shape[0:2]
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    if bounding_box_inside_image(img_array, bb):
        mask[bb[0]:(bb[0] + bb[2]), bb[1]:(bb[1] + bb[3])] = 1
    elif bb[0] + bb[2] <= w and bb[1] + bb[3] <= h:  # left and top edges are ok
        mask[bb[0]:min(bb[0] + bb[2], w), bb[1]:min(bb[1] + bb[3], h)] = 1
    else:  # left or top edge not ok so use entire box
        mask = np.ones((h, w), dtype=np.uint8)
    if mask.shape[0] != img_array.shape[0] or mask.shape[1] != img_array.shape[1]:
        print('trouble with mask size in bb_to_mask, resetting to image size')
        mask = np.ones((h, w), dtype=np.uint8)

    return mask


def is_valid_image(img_array):
    if img_array is not None and \
            type(img_array) == np.ndarray and\
            img_array.shape[0] * img_array.shape[1] >= constants.min_image_area:
        return True
    else:
        return False


def is_valid_local_image_file(img_filename):
    img_array = cv2.imread(img_filename)
    return is_valid_image(img_array)


def is_valid_local_or_remote_image_file(img_filename):
    img_array = get_cv2_img_array(img_filename)
    return is_valid_image(img_array)



def get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, convert_url_to_local_filename=False, download=False,
                      download_directory='images', filename=False, replace_https_with_http=True):
    """
    Get a cv2 img array from a number of different possible inputs.

    :param url_or_path_to_image_file_or_cv2_image_array:
    :param convert_url_to_local_filename:
    :param download:
    :param download_directory:
    :return: img_array
    """
    # print('get:' + str(url_or_path_to_image_file_or_cv2_image_array) + ' try local' + str(
    # convert_url_to_local_filename) + ' download:' + str(download))
    got_locally = False
    img_array = None  # attempt to deal with non-responding url

    # first check if we already have a numpy array
    if isinstance(url_or_path_to_image_file_or_cv2_image_array, np.ndarray):
        img_array = url_or_path_to_image_file_or_cv2_image_array

    # otherwise it's probably a string, check what kind
    elif isinstance(url_or_path_to_image_file_or_cv2_image_array, basestring):
        # try getting url locally by changing url to standard name
        if convert_url_to_local_filename:  # turn url into local filename and try getting it again
            # filename = url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[0]
            # jeremy changed this since it didn't work with url -
            # https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcR2oSMcnwErH1eqf4k8fvn2bAxvSdDSbp6voC7ijYJStL2NfX6v
            # TODO: find a better way to create legal filename from url
            filename = \
                url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                    -1]
            filename = os.path.join(download_directory, filename)
            if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('.bmp') or \
                    filename.endswith('tiff'):
                pass
            else:  # there's no 'normal' filename ending so add .jpg
                filename = filename + '.jpg'
            # print('trying again locally using filename:' + str(filename))
            img_array = get_cv2_img_array(filename, convert_url_to_local_filename=False, download=download,
                                          download_directory=download_directory)
            # maybe return(get_cv2 etc) instead of img_array =
            if img_array is not None:
                # print('got ok array calling self locally')
                return img_array
            else:  # couldnt get locally so try remotely
                # print('trying again remotely since using local filename didnt work, download=' + str( download) + ' fname:' + str(filename))
                return (
                    get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, convert_url_to_local_filename=False,
                                      download=download,
                                      download_directory=download_directory))  # this used to be 'return'
        # put images in local directory
        else:
            # get remotely if its a url, get locally if not
            if "://" in url_or_path_to_image_file_or_cv2_image_array:
                if replace_https_with_http:
                    url_or_path_to_image_file_or_cv2_image_array = url_or_path_to_image_file_or_cv2_image_array.replace(
                        "https", "http")
                img_url = url_or_path_to_image_file_or_cv2_image_array
                try:
                    # print("trying remotely (url) ")
                    headers = {'User-Agent': USER_AGENT}
                    response = requests.get(img_url, headers=headers)  # download
                    img_array = imdecode(np.asarray(bytearray(response.content)), 1)
                except ConnectionError:
                    logging.warning("connection error - check url or connection")
                    return None
                except:
                    logging.warning(" error other than connection error - check something other than connection")
                    return None

            else:  # get locally, since its not a url
                # print("trying locally (not url)")
                img_path = url_or_path_to_image_file_or_cv2_image_array
                try:
                    img_array = cv2.imread(img_path)
                    if img_array is not None:
                        # print("success trying locally (not url)")
                        got_locally = True
                    else:
                        # print('couldnt get locally (in not url branch)')
                        return None
                except:
                    # print("could not read locally, returning None")
                    logging.warning("could not read locally, returning None")
                    return None  # input isn't a basestring nor a np.ndarray....so what is it?
    else:
        logging.warning("input is neither an ndarray nor a string, so I don't know what to do")
        return None

    # After we're done with all the above, this should be true - final check that we're outputting a good array
    if not (isinstance(img_array, np.ndarray) and isinstance(img_array[0][0], np.ndarray)):
        print("Bad image coming into get_cv2_img_array - check url/path/array:" + str(
            url_or_path_to_image_file_or_cv2_image_array) + 'try locally' + str(
            convert_url_to_local_filename) + ' dl:' + str(
            download) + ' dir:' + str(download_directory))
        logging.warning("Bad image - check url/path/array:" + str(
            url_or_path_to_image_file_or_cv2_image_array) + 'try locally' + str(
            convert_url_to_local_filename) + ' dl:' + str(
            download) + ' dir:' + str(download_directory))
        return (None)
    # if we got good image and need to save locally :
    if download:
        if not got_locally:  # only download if we didn't get file locally
            if not os.path.isdir(download_directory):
                os.makedirs(download_directory)
            if "://" in url_or_path_to_image_file_or_cv2_image_array:  # its a url, get the bifnocho
                if replace_https_with_http:
                    url_or_path_to_image_file_or_cv2_image_array = url_or_path_to_image_file_or_cv2_image_array.replace(
                        "https", "http")
                filename = \
                    url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                        -1]
                filename = os.path.join(download_directory, filename)
            else:  # its not a url so use straight
                filename = os.path.join(download_directory, url_or_path_to_image_file_or_cv2_image_array)
            if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('.bmp') or filename.endswith(
                    'tiff'):
                pass
            else:  # there's no 'normal' filename ending
                filename = filename + '.jpg'
            try:  # write file then open it
                # print('filename for local write:' + str(filename))
                write_status = imwrite(filename, img_array)
                max_i = 50  # wait until file is readable before continuing
                gotfile = False
                for i in xrange(max_i):
                    try:
                        with open(filename, 'rb') as _:
                            gotfile = True
                    except IOError:
                        time.sleep(10)
                if gotfile == False:
                    print('Could not access {} after {} attempts'.format(filename, str(max_i)))
                    raise IOError('Could not access {} after {} attempts'.format(filename, str(max_i)))
            except:  # this is prob unneeded given the 'else' above
                print('unexpected error in Utils calling imwrite')
    return img_array


def count_human_bbs_in_doc(dict_of_images, skip_if_marked_to_skip=True):
    n = 0
    for entry in dict_of_images:
        print('entry:' + str(entry) + ' n=' + str(n), end='\r')
        if good_bb(entry, skip_if_marked_to_skip=skip_if_marked_to_skip):
            n = n + 1  # dont care if marked to be skipped
    return (n)


def average_bbs(bblist):
    avg_box = [0, 0, 0, 0]
    n = 0
    for bb in bblist:
        # print('avg'+str(avg_box))
        # print('bb'+str(bb))
        avg_box = np.add(avg_box, bb)
        # print('avg after'+str(avg_box))
        n = n + 1
    avg_box = np.int(np.divide(avg_box, n))
    return avg_box


def good_bb(dict, skip_if_marked_to_skip=True):
    '''
    determine if dict has good human bb in it
    '''

    if skip_if_marked_to_skip:
        if "skip_image" in dict:
            if dict['skip_image'] == True:
                return (False)

    if not 'url' in dict:
        # print('img is none')
        return (False)

    url = dict['url']
    img_arr = get_cv2_img_array(url, convert_url_to_local_filename=True, download=True,
                                download_directory='images')
    if not is_valid_image(img_arr):
        print('bad image array discovered in is_valid_image')
        return False
    if not 'human_bb' in dict:
        # print('no human_bb key in dict')
        return (False)
    if dict["human_bb"] is None:
        # print('human_bb is None')
        return (False)
    bb = dict['human_bb']
    if not bounding_box_inside_image(img_arr, bb):  #
        print('bad bb caught,bb:' + str(bb) + ' img size:' + str(img_arr.shape) + ' imagedoc:' + str(
            url))
        return (False)
    if all_inclusive_bounding_box(img_arr, bb):
        dict['human_bb'] = reduce_bounding_box(bb)  # attempting to avoid bbsize=imgsize
    return (True)


def legal_bounding_box(rect):
    if rect is None:
        return False
    minimum_allowed_area = constants.min_image_area
    if rect[2] * rect[3] < minimum_allowed_area:
        logging.warning('bb too small : area = ' + str(rect[2]) + 'x' + str(rect[3]) + ':' + str(rect[2] * rect[3]))
        return False
    if rect[0] < 0 or rect[1] < 0 or rect[2] < 0 or rect[3] < 0:
        return False
    return True


def bounding_box_inside_image(image_array, rect):
    # if check_img_array(image_array) and legal_bounding_box(rect):
    if legal_bounding_box(rect):
        height, width = image_array.shape[0:2]
        if rect[0] < width and rect[0] + rect[2] < width and rect[1] < height and rect[1] + rect[3] < height:
            return True  # bb fits into image
        else:
            #print('warning - bb not inside image')
            return False
    else:
        print('warning - bb not legal (either too small or None')
        return False


# products_collection_cursor = db.products.find()   #Regular db of one fig per item

# prefixes = ['Main Image URL angle ', 'Style Gallery Image ']
# training docs contains lots of different images (URLs) of the same clothing item
# logging.debug(str(doc))
# print('doc:'+str(doc))
#       for prefix in prefixes:


def fix_all_bbs_in_db(use_visual_output=False):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    if db is None:
        return {"success": 0, "error": "could not get db"}
    training_collection_cursor = db.training.find()
    print('returned cursor')
    assert (training_collection_cursor)  # make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    j = 0
    while doc is not None:
        print('doc:' + str(doc))
        images = doc['images']
        print('checking doc #' + str(j + 1))
        i = 0
        for image in images:
            image_url = image["url"]
            if 'skip_image' in image:
                if image['skip_image'] == True:
                    print('marked for skip:' + str(i), end='\r')
                    continue
            img_arr = get_cv2_img_array(image_url, convert_url_to_local_filename=True, download=True,
                                        download_directory='images')
            if not is_valid_image(img_arr):
                print('img is not valid (=None or too small')
                continue

            if 'human_bb' in image:
                i = i + 1
                height, width = img_arr.shape[0:2]
                bb = image["human_bb"]
                if bb is None:
                    print('bb is None')
                    continue

                cv2.rectangle(img_arr, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), color=[0, 0, 255],
                              thickness=2)
                cv2.imshow('img', img_arr)
                k = cv2.waitKey(50) & 0xFF
                if not bounding_box_inside_image(img_arr, bb):
                    print('bad bb caught,bb:' + str(bb) + ' img size:' + str(img_arr.shape) + ' imagedoc:' + str(
                        image) + ' h,w:' + str(height) + ',' + str(width))
                    print('h,w:' + str(height) + ',' + str(width))
                    if not legal_bounding_box(bb):  # too small, make right and bottom at edge of  image
                        print('not legal bounding box')
                        raw_input('not a legal bb...')
                        bb[2] = width - bb[0]
                        bb[3] = height - bb[1]
                    bb[0] = max(0, bb[0])  # if less than zero
                    bb[0] = min(bb[0], width - 1)  # if greater than width
                    bb[2] = max(0, bb[2])  # if less than 0
                    bb[2] = min(bb[2], width - bb[0] - 1)  # the -1 is just to make sure, prob unneeded

                    bb[1] = max(0, bb[1])  # if less than zero
                    bb[1] = min(bb[1], height - 1)  # if greater than height
                    bb[3] = max(0, bb[3])  # if less than zero
                    bb[3] = min(bb[3], height - bb[1] - 1)  # the -1 is just to make sure, prob unneeded
                    print('suggested replacement:' + str(bb))
                    raw_input('got one')
                    image["human_bb"] = bb
                    id = str(doc['_id'])
                    write_result = db.training.update({"_id": objectid.ObjectId(id)},
                                                      {"$set": {"images": doc['images']}})
                    # TODO: check error on updating
                    print('write result:' + str(write_result))
                else:
                    print('got good bb, i=' + str(i), end='\r', sep='')

        j = j + 1
        doc = next(training_collection_cursor, None)

    return {"success": 1}


def show_all_bbs_in_db(use_visual_output=True):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    if db is None:
        return {"success": 0, "error": "could not get db"}
    training_collection_cursor = db.training.find()
    print('returned cursor')
    assert (training_collection_cursor)  # make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    j = 0
    while doc is not None:
        print('doc:' + str(doc))
        images = doc['images']
        print('checking doc #' + str(j + 1))
        print(doc)
        i = 0
        for image in images:
            image_url = image["url"]
            if 'skip_image' in image:
                if image['skip_image'] == True:
                    print('marked for skip:' + str(i), end='\r')
                    continue
            img_arr = get_cv2_img_array(image_url, convert_url_to_local_filename=True, download=True,
                                        download_directory='images')
            if not is_valid_image(img_arr):
                print('img is not valid (=None or too small')
                continue

            if 'human_bb' in image:
                i = i + 1
                height, width = img_arr.shape[0:2]
                bb = image["human_bb"]
                if bb is None:
                    print('bb is None')
                    continue

                if not bounding_box_inside_image(img_arr, bb):
                    print('bad bb caught,bb:' + str(bb) + ' img size:' + str(img_arr.shape) + ' imagedoc:' + str(
                        image) + ' h,w:' + str(height) + ',' + str(width))

                    if use_visual_output:
                        # cv2.rectangle(img_arr, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), color=[0,255,0], thickness=2)
                        cv2.imshow('im1', img_arr)
                        k = cv2.waitKey(0) & 0xFF
                else:
                    print('got good bb, i=' + str(i), end='\r', sep='')

                    if use_visual_output:
                        cv2.rectangle(img_arr, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), color=[0, 255, 0],
                                      thickness=2)
                        cv2.imshow('im1', img_arr)
                        k = cv2.waitKey(0) & 0xFF
                        # raw_input('waiting for input')
        j = j + 1
        doc = next(training_collection_cursor, None)

    return {"success": 1}


def all_inclusive_bounding_box(image_array, bounding_box):
    """
    determine if the bb takes up all or  almost all the image
    :param image_array:
    :param bounding_box:
    :return:whether the bb takes up almost all image (True) or not (False)
    """
    height, width = image_array.shape[0:2]
    image_area = float(height * width)
    bb_area = bounding_box[2] * bounding_box[3]
    if bb_area > constants.min_bb_to_image_area_ratio * image_area:
        # print('got a bb that takes nearly all image')
        # logging.warning('got a bb that takes nearly all image')
        return True
    else:
        return False


def reduce_bounding_box(bounding_box):
    """
    determine if the bb takes up all or  almost all the image
    :param bounding_box:
    :return:smaller bb (again attempting to get around grabcut bug )
    """
    newx = bounding_box[0] + 1
    new_width = bounding_box[2] - 1
    newy = bounding_box[1] + 1
    new_height = bounding_box[3] - 1
    newbb = [newx, newy, new_width, new_height]
    if legal_bounding_box(newbb):
        return newbb
    else:
        logging.warning('cant decrease size of bb')
        return bounding_box



def image_stats_from_dir_of_dirs(dir_of_dirs,filter=None):
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    if filter is not None:
        only_dirs = [dir for dir in only_dirs if filter in dir]
    only_dirs.sort()
    hlist = []
    wlist = []
    dlist = []
    Blist = []
    Glist = []
    Rlist = []
    nlist = []
    n=0
    for a_dir in only_dirs:
        fulldir = os.path.join(dir_of_dirs,a_dir)
        print('analyzing dir '+fulldir)
        results = image_stats_from_dir(fulldir)
        if results is not None:
            hlist.append(results[0])
            wlist.append(results[1])
            dlist.append(results[2])
            Blist.append(results[3])
            Glist.append(results[4])
            Rlist.append(results[5])
            nlist.append(results[6])
            n += 1
    avg_h = np.average(hlist,weights=nlist)
    avg_w = np.average(wlist,weights=nlist)
    avg_d = np.average(dlist,weights=nlist)
    avg_B = np.average(Blist,weights=nlist)
    avg_G = np.average(Glist,weights=nlist)
    avg_R = np.average(Rlist,weights=nlist)
    totfiles = np.sum(nlist)
    print('weighted averages of {} directories: h:{} w{} d{} B {} G {} R {} totfiles {}'.format(n,avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles))
    return([avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles])

def image_chooser_dir_of_dirs(dir_of_dirs,dest_dir,removed_dir=None,filter=None,relabel_dir=None,multiple_dir=None):
    print('running images chooser source:{} dest:{} filter {}'.format(dir_of_dirs,dest_dir,filter))
    only_dirs = [d for d in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs, d))]
    if filter is not None:
        only_dirs = [d for d in only_dirs if filter in d]


    for d in only_dirs:
        actual_source = os.path.join(dir_of_dirs,d)
        actual_dest = os.path.join(dest_dir,d)

        if removed_dir is None:
            removed_dir = os.path.join(actual_source,'removed')
        if relabel_dir is None:
            relabel_dir = os.path.join(actual_source,'mislabelled')
        if multiple_dir is None:
            multiple_dir = os.path.join(actual_source,'multiple_items')

        Utils.ensure_dir(actual_dest)
        Utils.ensure_dir(removed_dir)
        Utils.ensure_dir(relabel_dir)
        Utils.ensure_dir(multiple_dir)
        image_chooser(actual_source,actual_dest,removed_dir=removed_dir,relabel_dir=relabel_dir,multiple_dir=multiple_dir)

def image_chooser(source_dir,dest_dir=None,removed_dir=None,relabel_dir=None,multiple_dir=None,ensure_jpg_suffix=True,remove_parens=True,display_size=(700,700)):

    print('starting image chooser source {} dest {}'.format(source_dir,dest_dir))
    if removed_dir is None:
        removed_dir = os.path.join(source_dir,'removed')
    if relabel_dir is None:
        relabel_dir = os.path.join(source_dir,'mislabelled')
    if multiple_dir is None:
        multiple_dir = os.path.join(source_dir,'multiple_items')
    if dest_dir is None:
        dest_dir = os.path.join(source_dir,'kept')
    Utils.ensure_dir(removed_dir)
    Utils.ensure_dir(multiple_dir)
    Utils.ensure_dir(relabel_dir)
    Utils.ensure_dir(dest_dir)
    print('choosing:'+str(source_dir)+'\ngood:'+str(dest_dir)+' \nremoved:'+str(removed_dir)+' \nreprocess:'+str(relabel_dir)+'\nmultiple:'+str(multiple_dir))
    only_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    n = len(only_files)
    if n==0:
        print('no files in '+source_dir)
        return
    i = -1
    elapsed_time=0
    tpi = 1
    alpha = 0.9
    time_done=time.time()
    n_deleted = 0
    n_kept = 1 #white lie to avoid /0
    while i < n-1 : #to allow undo need to manipulate index which doesnt work with iterator
        i = i + 1
        a_file = only_files[i]
        fullname = os.path.join(source_dir,a_file)
        if ensure_jpg_suffix:
            if a_file[-4:]!='.jpg':
                a_file=a_file+'.jpg'
        if remove_parens:
            a_file = a_file.replace('(','').replace(')','')

        img_arr = cv2.imread(fullname)

        if img_arr is  None:
            print('trouble gettting image '+fullname)
            continue
        shape = img_arr.shape
#            resized = img_arr
        resized = resize_keep_aspect(img_arr,output_size=display_size)
        h,w = img_arr.shape[0:2]
##            if h>200:
#               resized = cv2.resize(img_arr,(int((200.0*w)/h),200))
#               print('h,w {},{} newh neww {},{}'.format(h,w,resized.shape[0],resized.shape[1]))
        print('img '+str(i)+' of '+str(n)+':'+a_file+' shape:'+str(shape) +' (resized to '+str(resized.shape)+') kept:'+str(n_kept)+' deleted:'+str(n_deleted)+' %'+str(100*float(n_deleted)/n_kept))
        print('(q)uit (d)elete (k)eep (r)elabel (m)ultiple items (u)ndo tpi {}'.format(tpi))
        winname = a_file

        while(1):
            cv2.imshow(winname,resized)
            k = cv2.waitKey(0)

                # q to stop
            if k==ord('q'):
                print('quitting')
                sys.exit('quitting since you pressed q')
            elif k==ord('d'):  # normally -1 returned,so don't print it
#                print('removing '+a_file+' to '+removed_dir)
                dest_fullname = os.path.join(removed_dir,a_file)
                print('removing {}\nto {} '.format(fullname,dest_fullname))
                shutil.move(fullname, dest_fullname)
                prev_moved_to = dest_fullname
                prev_moved_from = fullname
                n_deleted = n_deleted + 1
                break
            elif k== ord('k'):
#                print('keeping '+a_file+' in '+dest_dir)
                dest_fullname = os.path.join(dest_dir,a_file)
                print('keeping {}\nto {} '.format(fullname,dest_fullname))
                shutil.move(fullname, dest_fullname)
                prev_moved_to = dest_fullname
                prev_moved_from = fullname
                n_kept = n_kept + 1
                break
            elif k== ord('r'):
                dest_fullname = os.path.join(relabel_dir,a_file)
                print('relabelling {}\nto {} '.format(fullname,dest_fullname))
                shutil.move(fullname, dest_fullname)
                prev_moved_to = dest_fullname
                prev_moved_from = fullname
                break
            elif k== ord('m'):
#                print('reprocessing '+a_file+' in '+reprocess_dir)
                dest_fullname = os.path.join(multiple_dir,a_file)
                print('multiple, moving {}\nto {} '.format(fullname,dest_fullname))
                shutil.move(fullname, dest_fullname)
                prev_moved_to = dest_fullname
                prev_moved_from = fullname
                break
            elif k== ord('u'):
#                print('reprocessing '+a_file+' in '+reprocess_dir)
                print('undo')
                shutil.move(prev_moved_to,prev_moved_from)
                i = i - 2
                break
            else:
                k = cv2.waitKey(0)
                print('unident key')
                #add 'back' option
        elapsed_time = time.time()-time_done
        tpi = alpha*tpi+(1-alpha)*elapsed_time
        time_done=time.time()
        cv2.destroyWindow(winname)

def image_stats_from_dir(dirname):
    only_files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    hlist = []
    wlist = []
    dlist = []
    Blist = []
    Glist = []
    Rlist = []
    n=0
    for filename in only_files:
        fullpath = os.path.join(dirname,filename)
        results = image_stats(fullpath)
        if results is not None:
    #        print(results)
            hlist.append(results[0])
            wlist.append(results[1])
            dlist.append(results[2])
            Blist.append(results[3])
            Glist.append(results[4])
            Rlist.append(results[5])
            n += 1
            print(str(n) +' of '+str(len(only_files)), end='\r')
            sys.stdout.flush()
    avg_h = np.mean(hlist)
    avg_w = np.mean(wlist)
    avg_d = np.mean(dlist)
    avg_B = np.mean(Blist)
    avg_G = np.mean(Glist)
    avg_R = np.mean(Rlist)
    print('dir:{} avg of {} images: h:{} w{} d{} B {} G {} R {}'.format(dirname,n,avg_h,avg_w,avg_d,avg_B,avg_G,avg_R))
    if n == 0 :
        return None
    return([avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,n])

def image_stats(filename):
    img_arr = cv2.imread(filename)
    if img_arr is not None:
        use_visual_output = False
        if(use_visual_output):
            cv2.imshow('current_fig',img_arr)
            cv2.waitKey(10)
        shape = img_arr.shape
        if len(shape)>2:   #BGR
            h=shape[0]
            w = shape[1]
            d = shape[2]
            avgB = np.mean(img_arr[:,:,0])
            avgG = np.mean(img_arr[:,:,1])
            avgR = np.mean(img_arr[:,:,2])
            return([h,w,d,avgB,avgG,avgR])
        else:  #grayscale /single-channel image has no 3rd dim
            h=shape[0]
            w=shape[1]
            d=1
            avgGray = np.mean(img_arr[:,:])
            return([h,w,1,avgGray,avgGray,avgGray])

    else:
        logging.warning('could not open {}'.format(filename))
        return None

def test_or_training_textfile(dir_of_dirs,test_or_train=None):
    '''
    takes dir of dirs each with different class, makes textfile suitable for training/test set
    :param dir_of_dirs:
    :return:
    '''
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    only_dirs.sort()
    print(str(len(only_dirs))+' dirs:'+str(only_dirs))
    if test_or_train:
        filename = os.path.join(dir_of_dirs,test_or_train+'.txt')
    else:
        filename = os.path.join(dir_of_dirs,'fileclasses.txt')
    with open(filename,'a') as myfile:  #append , don't clobber
        classno = 0
        for dir in only_dirs:
            if (not test_or_train) or dir[0:4]==test_or_train[0:4]:
                fulldir = os.path.join(dir_of_dirs,dir)
                print('fulldir:'+str(fulldir))
                only_files = [f for f in os.listdir(fulldir) if os.path.isfile(os.path.join(fulldir, f))]
                n = len(only_files)
                print('n files {} in {}'.format(n,dir))
                for a_file in only_files:
                    line = os.path.join(dir_of_dirs,dir, a_file) + ' '+ str(classno) + '\n'
                    myfile.write(line)
                classno += 1

def resize_and_crop_image( input_file_or_np_arr, output_file=None, output_side_length = 256,use_visual_output=False):
    '''Takes an image name, resize it and crop the center square
    '''
    #TODO - implement nonsquare crop
    if isinstance(input_file_or_np_arr,basestring):
        input_file_or_np_arr = cv2.imread(input_file_or_np_arr)
    height, width, depth = input_file_or_np_arr.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    resized_img = cv2.resize(input_file_or_np_arr, (new_width, new_height))
    height_offset = int((new_height - output_side_length) / 2)
    width_offset = int((new_width - output_side_length) / 2)
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
                              width_offset:width_offset + output_side_length]
    if use_visual_output is True:
        cv2.imshow('cropped', cropped_img)
        cv2.imshow('orig',input_file_or_np_arr)
        cv2.waitKey(0)
    if output_file is not None:
        cv2.imwrite(output_file, cropped_img)
    return cropped_img

def resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True):
    '''
    resizes to a maximum sidelength keeping orig. aspect ratio
    :param img_arr:
    :param max_sidelength:
    :param use_visual_output:
    :return:resized image
    '''
    h,w,c = img_arr.shape
    if h>w:
        #if h>max_sidelength:

        new_h = max_sidelength
        new_w = int(w*float(max_sidelength)/h)
        img_arr=cv2.resize(img_arr,(new_w,new_h))
        print('hshape ')
        print(img_arr.shape)
       # else:  #longest side is still under limit , show orig without resize
       #     pass
    else:
     #   if w>max_sidelength:
        new_w = max_sidelength
        new_h = int(h*float(max_sidelength)/w)
        img_arr=cv2.resize(img_arr,(new_w,new_h))
        print('shape')
        print(img_arr.shape)
     #   else:  #longest side is still under limit , show orig without resize
     #       pass
    if (use_visual_output):
        cv2.imshow('image',img_arr)
        cv2.waitKey(0)
    return img_arr

def resize_keep_aspect_dir(dir,outdir=None,overwrite=False,output_size=(256,256),use_visual_output=False,filefilter='.jpg',
                           careful_with_the_labels=False,recursive=False,kill_parens=True):
    '''
    you can avoid use of filter by specifying filefilter=''
    :param dir:
    :param outdir:
    :param overwrite:
    :param output_size:
    :param use_visual_output:
    :param filefilter:
    :param careful_with_the_labels:
    :param recursive:
    :return:
    '''
    if recursive:
        allfiles = []
        for root,dirs,files in os.walk(dir):
            path = root.split(os.sep)
#            print('root {}\ndirs {} '.format(root,dirs))
            allfiles = allfiles + [os.path.join(root,f) for f in files if filefilter in f]
 #       raw_input('ret to cont')
        files = allfiles
    else:
        files = [ os.path.join(dir,f) for f in os.listdir(dir) if filefilter in f]
    print(str(len(files))+' files in '+dir)
    for file in files:
#        fullname = os.path.join(dir,file)
        if overwrite:
            newname = file
        else:
            filebase = os.path.basename(file)
            basedir = os.path.dirname(file)
#            print('file {}\nbase {}\nalone {}'.format(file,basedir,filebase))
            if outdir:
                Utils.ensure_dir(outdir)
                newname = os.path.join(outdir,filebase)
            else:
                newname = filebase.split(filefilter)[0]+'_'+str(output_size[0])+'x'+str(output_size[1])+filefilter
                newname = os.path.join(basedir,newname)
        if kill_parens:
            newname=newname.replace('(','') #
            newname=newname.replace(')','')
        print('infile:{}\ndesired size:{}\noutfile {}'.format(file,output_size,newname))
#        raw_input('ret to cont')
        resize_keep_aspect(file, output_file=newname, output_size = output_size,use_visual_output=use_visual_output,careful_with_the_labels=careful_with_the_labels)

def resize_keep_aspect(input_file_or_np_arr, output_file=None, output_size = (256,256),use_visual_output=False,careful_with_the_labels=False, copy_edge_pixeles=False):
    '''
    Takes an image name/arr, resize keeping aspect ratio, filling extra areas with edge values
    :param input_file_or_np_arr:
    :param output_file:name for output
    :param output_size:size of output image (height,width)
    :param use_visual_output:
    :return:
    '''

    if isinstance(input_file_or_np_arr,basestring):
        input_file_or_np_arr = cv2.imread(input_file_or_np_arr)

    if input_file_or_np_arr is None:
        logging.warning('got a bad image')
        return
    inheight, inwidth = input_file_or_np_arr.shape[0:2]
    if inheight < 2 or inwidth < 2 : #1-pixel wide or high images cant be resized. actually just failed on a 6x7756 image->256x256...
        logging.warning('got a zero height or width imge {}'.format(input_file_or_np_arr.shape))
        return
    outheight, outwidth = output_size[:]
    if outheight == 0 or outwidth == 0:
        logging.warning('got a zero height or resize request {}'.format(output_size))
        return
    if inheight==outheight and inwidth==outwidth:
        logging.info('resize not needed , in {}x{} out {}x{}'.format(inheight,inwidth,outheight,outwidth))
        return input_file_or_np_arr
    logging.info('doing resize , input hxw  {} {} output hxw {} {}'.format(inheight,inwidth,outheight,outwidth))
    if inheight == 0 or inwidth == 0:
        logging.warning('got a bad image')
        return
    out_ar = float(outheight)/outwidth
    in_ar = float(inheight)/inwidth
    if len(input_file_or_np_arr.shape) == 3:
        indepth = input_file_or_np_arr.shape[2]
        output_img = np.zeros([outheight,outwidth,indepth],dtype=np.uint8)
    else:
        indepth = 1
        output_img = np.zeros([outheight,outwidth],dtype=np.uint8)
#    print('input:{}x{}x{}'.format(inheight,inwidth,indepth))
    actual_outheight, actual_outwidth = output_img.shape[0:2]
#    print('output:{}x{}'.format(actual_outheight,actual_outwidth))
    if out_ar < in_ar:  #resize height to output height and fill left/right
        factor = float(inheight)/outheight
        new_width = int(float(inwidth) / factor)
        try:
            resized_img = cv2.resize(input_file_or_np_arr, (new_width, outheight))
        except:
            e = sys.exc_info()[0]
            logging.warning('error on resizing {} to {} error:{}'.format(input_file_or_np_arr.shape,output_size,e))
 #           raw_input('ret to cont')
            return
 #       print('<resize size:'+str(resized_img.shape)+' desired width:'+str(outwidth)+' orig width resized:'+str(new_width))
        width_offset = (outwidth - new_width ) / 2
        logging.debug('output ar<  input ar , width padding around '+str(width_offset)+ ' to '+str(width_offset+new_width))
        output_img[:,width_offset:width_offset+new_width] = resized_img[:,:]
        for n in range(0,width_offset):  #doing this like the below runs into a broadcast problem which could prob be solved by reshaping
#            output_img[:,0:width_offset] = resized_img[:,0]
#            output_img[:,width_offset+new_width:] = resized_img[:,-1]
            if copy_edge_pixeles:
                output_img[:,n] = resized_img[:,0]
                output_img[:,n+new_width+width_offset] = resized_img[:,-1]
    else:   #resize width to output width and fill top/bottom
        factor = float(inwidth)/outwidth
        new_height = int(float(inheight) / factor)
        try:
            resized_img = cv2.resize(input_file_or_np_arr, (outwidth, new_height))
        except:
            e = sys.exc_info()[0]
            logging.warning('error on resizing {} to {} error:{}'.format(input_file_or_np_arr.shape,output_size,e))
#            raw_input('ret to cont')
            return

        height_offset = (outheight - new_height) / 2
        logging.debug('output ar >=  input ar , height padding around '+str(height_offset)+' to '+str(height_offset+new_height))
        output_img[height_offset:height_offset+new_height,:] = resized_img[:,:]
        if copy_edge_pixeles:
            output_img[0:height_offset,:] = resized_img[0,:]
            output_img[height_offset+new_height:,:] = resized_img[-1,:]
#        print('resize size:'+str(resized_img.shape)+' desired height:'+str(outheight)+' orig height resized:'+str(new_height))
#        print('orig dims {} resized to {}'.format(input_file_or_np_arr.shape,output_img.shape))

    if careful_with_the_labels:
        #kill any extranneous labels that have popped up
#        print('uniques in source:'+str(np.unique(input_file_or_np_arr)))
#        print('uniques in dest:'+str(np.unique(output_img)))
        for u in np.unique(output_img):
            if not u in input_file_or_np_arr: #
#                print('found new val in target:'+str(u))
                output_img[output_img==u] = 0
#        print('uniques in dest:'+str(np.unique(output_img)))
        # assert((np.unique(output_img)==np.unique(input_file_or_np_arr)).all())   this fails , bool attr has no all()
    if use_visual_output is True:
        cv2.imshow('output', output_img)
        cv2.imshow('orig',input_file_or_np_arr)
#        cv2.imshow('res',resized_img)
        cv2.waitKey(0)
    if output_file is not None:
        cv2.imwrite(output_file, output_img)
    return output_img
#dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

def resize_by_adding_border(img_arr,output_size,visual_output=False,copy_edge_pixels=False):
    img_arr = get_cv2_img_array(img_arr)
    if img_arr.shape[0]>output_size[0] or img_arr.shape[1]>output_size[1]:
        logging.warning('image to be bordered larger than requested size')
        img_arr = resize_keep_aspect(img_arr,output_size=output_size)
        return img_arr #this is not really what was asked for but it keeps the peace
    border_sizes = (output_size[0] - img_arr.shape[0],output_size[1] - img_arr.shape[1])
    new_image = np.zeros([output_size[0],output_size[1],3],dtype=np.uint8)
    top = border_sizes[0]/2
    bottom = top + img_arr.shape[0]
    left = border_sizes[1]/2
    right = left + img_arr.shape[1]
    new_image[top:bottom,left:right] = img_arr
    logging.debug('top {} bot {} d {} l {} r {} d {} imgarr {} '.format(top,bottom,bottom-top,left,right,right-left,img_arr.shape))
    if visual_output:
        cv2.imshow('resized from {} to {}'.format(img_arr.shape,new_image.shape),new_image)
        cv2.imshow('orig',img_arr)
        cv2.waitKey(0)
    return(new_image)

def undo_resize_keep_aspect(input_file_or_np_arr, output_file=None, output_size = (256,256),use_visual_output=False,careful_with_the_labels=False):
    '''
    Takes an image name/arr, resize keeping aspect ratio, filling extra areas with edge values
    :param input_file_or_np_arr:
    :param output_file:name for output
    :param output_size:size of output image (height,width)
    :param use_visual_output:
    :return:
    '''

    if isinstance(input_file_or_np_arr,basestring):
        input_file_or_np_arr = cv2.imread(input_file_or_np_arr)

    if input_file_or_np_arr is None:
        logging.warning('got a bad image')
        return
    #the shoe is on the other foot.
    inheight, inwidth = output_size[:]
    outheight, outwidth = input_file_or_np_arr.shape[0:2]
    logging.info('undoing resize , original hxw  {} {} resized hxw {} {}'.format(inheight,inwidth,outheight,outwidth))
    if (inheight == 0) or (inwidth == 0):
        logging.warning('got a bad image')
        return
    original_ar = float(inheight)/inwidth
    resized_ar = float(outheight)/outwidth
    if len(input_file_or_np_arr.shape) == 3:
        indepth = input_file_or_np_arr.shape[2]
        output_img = np.ones([outheight,outwidth,indepth],dtype=np.uint8)
    else:
        indepth = 1
        output_img = np.ones([outheight,outwidth],dtype=np.uint8)
#    print('input:{}x{}x{}'.format(inheight,inwidth,indepth))
    actual_outheight, actual_outwidth = output_img.shape[0:2]
#    print('output:{}x{}'.format(actual_outheight,actual_outwidth))
    if original_ar > resized_ar:  #unfil left/right and resize height to output height
        factor = float(inheight)/outheight
        new_width = int(float(inwidth) / factor)
        width_offset = (outwidth - new_width ) / 2
        remainder = outwidth - width_offset
        logging.debug('orig ar>  resized ar , width padding '+str(width_offset)+', taking from padding to '+str(remainder))
        output_img = input_file_or_np_arr[:,width_offset:remainder]
#        output_img[:,width_offset:width_offset+new_width] = resized_img[:,:]
        output_img = cv2.resize(output_img, (output_size[1],output_size[0]))  #cv2 does wxh not hxw
        #print('>resize size:'+str(output_img.shape))
    else:   #resize width to output width and fill top/bottom
        factor = float(inwidth)/outwidth
        new_height = int(float(inheight) / factor)
        height_offset = (outheight - new_height) / 2
        remainder = outheight - height_offset
        logging.debug('orig ar <=  resized ar , height padding '+str(height_offset)+ ',filling to '+str(remainder)+' outsize:'+str(output_size))
        output_img = input_file_or_np_arr[height_offset:remainder,:]
        #print('intermediate outputsize:'+str(output_img.shape))
        output_img = cv2.resize(output_img, (output_size[1],output_size[0])) #cv2 does wxh not hxw
        logging.debug('resize size:'+str(output_img.shape))
#        print('orig dims {} resized to {}'.format(input_file_or_np_arr.shape,output_img.shape))

    if careful_with_the_labels:
        #todo - the real way to do this is break into n channels and resize each individually
        #this may possibly be done by putting a loop over channels and calling this function recursively n_chan times
        #kill any extranneous labels that have popped up
#        print('uniques in source:'+str(np.unique(input_file_or_np_arr)))
#        print('uniques in dest:'+str(np.unique(output_img)))
        for u in np.unique(output_img):
            if not u in input_file_or_np_arr: #
#                print('found new val in target:'+str(u))
                output_img[output_img==u] = 0
#        print('uniques in dest:'+str(np.unique(output_img)))
        # assert((np.unique(output_img)==np.unique(input_file_or_np_arr)).all())   this fails , bool attr has no all()
    if use_visual_output is True:
        cv2.imshow('output', output_img)
        cv2.imshow('orig',input_file_or_np_arr)
#        cv2.imshow('res',resized_img)
        cv2.waitKey(0)
    if output_file is not None:
        cv2.imwrite(output_file, output_img)
    return output_img
#dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

def mask_to_rects(mask, visual_output=False, labels=constants.ultimate_21):
    '''
    given mask (eg from pixel level, not binary but several discrete values),
    find boudning boxes for 'reasonably large' blobs, maybe return just one per mask value ?
    :param mask:
    :return:
    '''
    uniques = np.unique(mask)
    if visual_output:
        show_mask_with_labels(mask,labels=labels,visual_output=True)
    bbs={}
    for u in uniques :
        if u == 0 :
            continue #not intstd in bgnd
        if 'skin' in labels:
            if u==labels.index('skin'):
                continue #not intstd in skin
        img = np.array((mask==u)*255,dtype=np.uint8)
        if len(img.shape)==3:
            print('got multichan image , taking 1st')
            img = img[:,:,0] #take first channel;
        n_pixels = np.shape(np.where(img!=0))[1]  #length of 1xn vector
        print('size of mask=={} is {} (shape {})'.format(u,n_pixels,np.shape(np.where(img!=0))[1]))
        if 0:
            # thismask = img*255
            # show_mask_with_labels(thismask,labels=constants.ultimate_21,visual_output=True)
            cv2.imshow("mask=={}".format(u), img)
            cv2.waitKey(0)

#        contours = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE,contours)
  #      contours = cv2.findContours(im2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        im2, contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #igc.copy() required here , this seems like it must be a cv2 bug
#        print('contours for {}: {}'.format(u,conts))
      #  cv2.drawContours(img*255,contours,-1,0,01)
      #   cv2.drawContours(im2,contours,-1,(100,255,100),5)
      #   cv2.imshow('contours',im2)
      #   cv2.waitKey(0)

        print('n contours:'+str(len(contours)))
        min_area_size = 1000  #this is kind of arbitrary , trying to keep out small blobs
        n_contour = 0
        im3 = np.zeros_like(img)
        max_area = 0
        next_area = 0
        n_max = -1
        n_next = -1
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
            area  = cv2.contourArea(cnt)
            if area > max_area:  #instead of just keeping max one could also try to bound all contours
                next_area=max_area
                n_next = n_max
                max_area=area
                n_max = n_contour
            if  area > min_area_size : #and cv2.isContourConvex(cnt):
                print('contour area of contour  {} is {}'.format(n_contour,area))
                if visual_output:
                    cv2.drawContours(im3,contours,n_contour,(50,255,50),2)
#                    cv2.imshow('current contour',im3)
            n_contour+=1

        if visual_output:
            cv2.imshow('big contours',im3)
            cv2.waitKey(0)

        cv2.drawContours(im3,contours,n_max,(244,100,150),5)
        x,y,w,h = cv2.boundingRect(contours[n_max])
#        x,y,w,h=None,None,None,None
        if visual_output :
            cv2.rectangle(im3,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.imshow('the biggest contour(s)',im3)
            cv2.waitKey(0)
        print('contour {} is biggest at len {}, {} is second at {}'.format(n_max,max_area,n_next,next_area))
        if max_area>min_area_size:
            bbs[labels[u]] = [x,y,w,h]
    return(bbs)

def resize_and_crop_maintain_bb( input_file_or_np_arr, output_file=None, output_width = 150, output_height = 200,use_visual_output=False,bb=None):
    '''Takes an image name, resize it and crop the center square
    '''
    #TODO - implement nonsquare crop
    #done
    #TODO - implement non-square resize up to maximum deformation e.g. 10% xscale=2 yscale=2.2
    if isinstance(input_file_or_np_arr,basestring):
        print('got image name '+str(input_file_or_np_arr))
        if bb is None:
            if 'bbox_' in input_file_or_np_arr:
                strs = input_file_or_np_arr.split('bbox_')
                bb_str = strs[1]
                coords = bb_str.split('_')
                bb_x = int(coords[0])
                bb_y = int(coords[1])
                bb_w = int(coords[2])
                bb_h = coords[3].split('.')[0]  #this has .jpg or .bmp at the end
                bb_h = int(bb_h)
                bb=[bb_x,bb_y,bb_w,bb_h]
                if bb_h == 0:
                    logging.warning('bad height encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
                    return None
                if bb_w == 0:
                    logging.warning('bad width encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
                    return None
        input_file_or_np_arr_name = input_file_or_np_arr
        input_file_or_np_arr = cv2.imread(input_file_or_np_arr)
        if input_file_or_np_arr is None:
            logging.warning('input file {} is none'.format(input_file_or_np_arr_name))
            return None
    img_height, img_width, img_depth = input_file_or_np_arr.shape

    if bb is None:
        bb = [0,0, img_width,img_height]
        print('no bbox given, using entire image')
    print('bb (x,y,w,h) {} {} {} {} image:{}x{} desired:{}x{}'.format(bb[0],bb[1],bb[2],bb[3],img_width,img_height,output_width,output_height))
    if bb[0]<0:
        logging.warning('BB x out of bounds, being reset')
        bb[0]=0
    if bb[1]<0 :
        bb[1]=0
        logging.warning('BB y out of bounds, being reset')
    if bb[0]+bb[2] > img_width:
        logging.warning('BB width out of bounds, being reset')
        bb[2]=img_width-bb[0]
    if bb[1]+bb[3] > img_height:
        logging.warning('BB height out of bounds, being reset')
        bb[3]=img_height - bb[1]

    orig_bb = copy.deepcopy(bb)
    in_aspect = float(img_width)/img_height
    out_aspect = float(output_width)/output_height
    width_out_in_ratio = float(output_width)/img_width
    height_out_in_ratio = float(output_height)/img_height
    if width_out_in_ratio > height_out_in_ratio:  #rescale by smallest amt possible
#    if abs(1-width_out_in_ratio) < abs(1-height_out_in_ratio):  #rescale by smallest amt possible
 #   if output_width >  output_height:  #rescale by smallest amt possible
        #this may be wrong when width_input>1 and height_inout<1 or vice versa
        new_width = int(width_out_in_ratio*img_width)  #should be output_width.  try round instead of int, didnt work
        new_height = int(width_out_in_ratio*img_height)  #may besomething other than output_height
        bb = np.multiply(bb,width_out_in_ratio)
        bb = [int(i) for i in bb]
        print('match width, new w,h:{},{} new bb {},{},{},{}'.format(new_width,new_height,bb[0],bb[1],bb[2],bb[3]))
        scaled_img = cv2.resize(input_file_or_np_arr,(new_width,new_height))
        y1 = bb[1]
        y2 = bb[1] + bb[3]

        height_to_crop = new_height - output_height
        output_extra_margin_over_bb = int(float(new_height-output_height )/2)
        ymin = y1 - output_extra_margin_over_bb

        print('tentative ymin '+str(ymin)+' extra margin '+str(output_extra_margin_over_bb))
        if ymin<0:
            ymin = 0
#            ymax = bb[3]
            ymax = output_height
        else:
            ymax = y2 + output_extra_margin_over_bb
            if ymax>new_height:
                ymax = new_height
#                ymin = ymax - bb[3]
                ymin = new_height-output_height
        print('new ymin,ymax:{},{}'.format(ymin,ymax))
        cropped_img = scaled_img[ymin:ymax,0:output_width,:]   #crop image
        bb[1] = bb[1]-ymin

    else:  #matching output height, width should be more than desired
        new_width = int(height_out_in_ratio*img_width)  #maybe other
        new_height = int(height_out_in_ratio*img_height)  #should be output_height
        bb = np.multiply(bb,height_out_in_ratio)
        bb = [int(i) for i in bb]
        print('match height, new w,h:{},{} new bb {},{},{},{}'.format(new_width,new_height,bb[0],bb[1],bb[2],bb[3]))
        scaled_img = cv2.resize(input_file_or_np_arr,(new_width,new_height))

        x1 = bb[0]
        x2 = bb[0] + bb[2]

        width_to_crop = new_width - output_width
        output_extra_margin_over_bb = int(float(new_width-output_width)/2)
        bb_center_x
        xmin = x1 - output_extra_margin_over_bb
        print('tentative xmin '+str(xmin)+' extra margin '+str(output_extra_margin_over_bb))
        if xmin<0:
            xmin = 0
#            xmax = bb[2]
            xmax = output_width
        else:
            xmax = x2 + output_extra_margin_over_bb

            if xmax>new_width:
                xmax = new_width
                xmin = new_width-output_width
        print('new xmin,xmax:{},{}'.format(xmin,xmax))
        cropped_img = scaled_img[0:output_height,xmin:xmax,:]
        bb[0] = bb[0]-xmin

    raw_input('enter to continue')

    if use_visual_output is True:
        cropped_copy = copy.deepcopy(cropped_img)
        cv2.rectangle(cropped_copy,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),color=[0,255,0 ])
        cv2.imshow('scaled_cropped', cropped_copy)
        orig_copy = copy.deepcopy(input_file_or_np_arr)
        cv2.rectangle(orig_copy,(orig_bb[0],orig_bb[1]),(orig_bb[0]+orig_bb[2],orig_bb[1]+orig_bb[3]),color=[0,255,0 ])

        cv2.imshow('orig',orig_copy)
        cv2.waitKey(0)
    if output_file is  None:
        if input_file_or_np_arr_name:
            output_file = orig_copy
        print('writing to:'+output_file)
        retval = cv2.imwrite(output_file, cropped_img)
        if retval is False:
             logging.warning('retval from imwrite is false (attempt to write file:'+output_file+' has failed :(  )')
    return cropped_img

def resize_and_crop_image_using_bb( input_file_or_np_arr, bb=None, output_file=None, output_w = 128,output_h = 128,use_visual_output=False):
    '''Takes an image name, resize it and crop the bb area, keeping as much of orig as possible
    '''
    #TODO - implement nonsquare crop
    # done
    if isinstance(input_file_or_np_arr,basestring):
        orig_name = input_file_or_np_arr
        input_file_or_np_arr = cv2.imread(input_file_or_np_arr)
        if input_file_or_np_arr is None:
            logging.debug('trouble reading input file {}'.format(orig_name))
            return
        if 'bbox_' in orig_name and bb is None:
            strs = orig_name.split('bbox_')
            bb_str = strs[1]
            coords = bb_str.split('_')
            bb_x = int(coords[0])
            bb_y = int(coords[1])
            bb_w = int(coords[2])
            bb_h = coords[3].split('.')[0]  #this has .jpg or .bmp at the end
            bb_h = int(bb_h)
            bb=[bb_x,bb_y,bb_w,bb_h]
            print('bb:'+str(bb))
            if bb_h == 0:
                logging.warning('bad height encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
                return None
            if bb_w == 0:
                logging.warning('bad width encountered in imutils.resize_and_crop_image for '+str(input_file_or_np_arr))
                return None

    height, width, depth = input_file_or_np_arr.shape

    if bb is None:
        bb = [0,0, width,height]
        print('no bbox given, using entire image')


    in_aspect = float(bb[2])/bb[3]
    out_aspect = float(output_w)/output_h
    x1 = bb[0]
    x2 = bb[0] + bb[2]
    y1 = bb[1]
    y2 = bb[1] + bb[3]
    if in_aspect>out_aspect:
        extra_pad_y = int((output_h*bb[2]/output_w - bb[3]) / 2)
        round = (output_h*bb[2]/output_w - bb[3]) % 2
        y1 = max(0,bb[1] - extra_pad_y)
        y2 = min(height,bb[1]+bb[3]+extra_pad_y+round)
        #print('pad y {} y1 {} y2 {}'.format(extra_pad_y,y1,y2))
    elif in_aspect<out_aspect:
        extra_pad_x = int((output_w*bb[3]/output_h - bb[2]) / 2)
        round = (output_w*bb[3]/output_h - bb[2]) % 2
        x1 = max(0,bb[0] - extra_pad_x)
        x2 = min(width,bb[0]+bb[2]+extra_pad_x+round)
        #print('pad x {} x1 {} x2 {}'.format(extra_pad_x,x1,x2))
    #print('x1 {} x2 {} y1 {} y2 {}'.format(x1,x2,y1,y2))
    cropped_img = input_file_or_np_arr[y1:y2,x1:x2,:]

    logging.debug('orig size {}x{} cropped to:{}x{},ar={} desired ar={}'.format(input_file_or_np_arr.shape[0],input_file_or_np_arr.shape[1],cropped_img.shape[0],cropped_img.shape[1],float(cropped_img.shape[1])/cropped_img.shape[0],float(output_w)/output_h))
    scaled_cropped_img = cv2.resize(cropped_img,(output_w,output_h))
#    print('resized to : {}x{}, ar={}, desired ar={}'.format(scaled_cropped_img.shape[0],scaled_cropped_img.shape[1],float(scaled_cropped_img.shape[1])/scaled_cropped_img.shape[0],float(output_w/output_h)))
    if use_visual_output is True:
        cv2.imshow('scaled_cropped', scaled_cropped_img)
        scaled_input = cv2.resize(input_file_or_np_arr,(output_w,output_h))
        cv2.imshow('orig',scaled_input)
        cv2.waitKey(0)
    if output_file is not None:
#        orig_dir = os.path.dirname(orig_name)
  #      orig_name_only = os.path.basename(orig_name)
    #    output_file = os.path.join(orig_dir,output_dir)
        print('writing to:'+output_file)
        retval = cv2.imwrite(output_file, scaled_cropped_img)
        if retval is False:
             logging.warning('retval from imwrite is false (attempt to write file:'+output_file+' has failed :(  )')
    return scaled_cropped_img

def center_crop(input_file_or_np_arr,cropsize):
    img_arr = Utils.get_cv2_img_array(input_file_or_np_arr)
    if img_arr is None:
        print('couldnt get img arr in imutils.center_crop')
        return
    h,w = img_arr.shape[0:2]
    if cropsize[0]>h or cropsize[1]>w:
        print('cropsize {} > imagesize {}'.format(cropsize,img_arr.shape))
        return
    h_margin = (h-cropsize[0])/2
    w_margin = (w-cropsize[1])/2
    out_arr = img_arr[h_margin:h_margin+cropsize[0],w_margin:w_margin+cropsize[1]] #takes care of odd h-crop[0]
    return out_arr

def resize_and_crop_maintain_aspect(img_arr_or_url,resize_size,crop_size):
    resized = resize_keep_aspect(img_arr_or_url,output_size=resize_size)
    cropped = center_crop(resized,crop_size)
    return cropped

def crop_files_in_dir(dirname,save_dir,**arglist):
    '''
    takes a function that has a filename as first arg and maps it onto files in dirname
    :param func: function to map
    :param dirname: dir of files to do function on
    :param arglist: args to func
    :return:
    '''
    Utils.ensure_dir(save_dir)
    logging.debug('cropping files in directory {} with arguments {}'.format(dirname,str(arglist)))
    only_files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    for a_file in only_files:
        input_path = os.path.join(dirname,a_file)
        output_path = os.path.join(save_dir,a_file)
        arglist['output_file']=output_path
        resize_and_crop_image_using_bb(input_path,**arglist)

def crop_files_in_dir_of_dirs(dir_of_dirs,**arglist):
    '''
    takes a function that has a filename as first arg and maps it onto files in directory of directories
    :param func: function to map
    :param dir_of_dirs: dir of dirs to do function on
    :param arglist: args to func
    :return:
    '''
    logging.debug('cropping files in directories under directory {} with arguments {}'.format(dir_of_dirs,str(arglist)))
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    num_cores = multiprocessing.cpu_count()
    fullpaths = []
    save_dirs = []
    for a_dir in only_dirs:
        fullpath = os.path.join(dir_of_dirs,a_dir)
        save_dir =  os.path.join(dir_of_dirs,'cropped/')
        save_dir =  os.path.join(save_dir,a_dir)
        Utils.ensure_dir(save_dir)
        fullpaths.append(fullpath)
        save_dirs.append(save_dir)
        crop_files_in_dir(fullpath,save_dir,**arglist)

# this will work if i can find how to do [x,y for x in a for y in b] 'zip' style
#     Parallel(n_jobs=num_cores)(delayed(crop_files_in_dir)(the_dir,the_path) for the_dir, the_path  in [fullpaths,save_dirs])

def kill_the_missing(sourcedir, targetdir):
    '''
    this removes anything not in the source , from the target
    :param sourcedir: has files removed relative to target
    :param targetdir: has extra files, we want to remove the extras it has relative to source
    :return:
    '''
    files_in_source = [f for f in os.listdir(sourcedir) if os.path.isfile(os.path.join(sourcedir,f))]
    files_in_target = [f for f in os.listdir(targetdir) if os.path.isfile(os.path.join(targetdir,f))]
    print('{} files in {}, {} files in {}'.format(len(files_in_source),sourcedir,len(files_in_target),targetdir))
    kill_dir = os.path.join(targetdir,'removed')
    Utils.ensure_dir(kill_dir)
    n_matched = 0
    n_killed = 0
    for a_file in files_in_target:
        if a_file in files_in_source:
            print('file {} in both dirs'.format(a_file))
            n_matched += 1
        else:
            print('file {} not matched, moving to {}'.format(a_file,kill_dir))
            shutil.move(os.path.join(targetdir,a_file), os.path.join(kill_dir,a_file))
            n_killed += 1
    print('n matched {} n killed {}'.format(n_matched,n_killed))
    files_in_source = [f for f in os.listdir(sourcedir) if os.path.isfile(os.path.join(sourcedir,f))]
    files_in_target = [f for f in os.listdir(targetdir) if os.path.isfile(os.path.join(targetdir,f))]
    print('{} files in {}, {} files in {}'.format(len(files_in_source),sourcedir,len(files_in_target),targetdir))

def find_the_common(sourcedir, targetdir):
    '''
    this removes anything not in the source , from the target
    :param sourcedir: has files removed relative to target
    :param targetdir: has extra files, we want to remove the extras it has relative to source
    :return:
    '''
    files_in_source = [f for f in os.listdir(sourcedir) if os.path.isfile(os.path.join(sourcedir,f))]
    files_in_target = [f for f in os.listdir(targetdir) if os.path.isfile(os.path.join(targetdir,f))]
    print('{} files in {}, {} files in {}'.format(len(files_in_source),sourcedir,len(files_in_target),targetdir))
    n_matched = 0
    n_not_matched = 0
    for a_file in files_in_target:
        if a_file in files_in_source:
            print('file {} in both dirs'.format(a_file))
            n_matched += 1
        else:
            print('file {} not matched'.format(a_file))
            n_not_matched += 1
    print('n matched {} n not matched {}'.format(n_matched,n_not_matched))
    files_in_source = [f for f in os.listdir(sourcedir) if os.path.isfile(os.path.join(sourcedir,f))]
    files_in_target = [f for f in os.listdir(targetdir) if os.path.isfile(os.path.join(targetdir,f))]

def oversegment(img_arr):
    image_height,image_width,image_channels = img_arr.shape
    num_superpixels = 100
    num_levels = 20
    cv2.SuperpixelSEEDS.createSuperpixelSEEDS(image_width, image_height, image_channels, num_superpixels, num_levels, use_prior = 2, histogram_bins=5, double_step = False)

def defenestrate_labels(mask,kplist):
    matches = np.ones_like(mask)
    for i in range(0,len(kplist)):
        index = kplist[i]
        nv = np.multiply(mask == index,i)
        print(nv.shape)
        matches = np.add(matches,nv)
    return matches

def defenestrate_directory(indir, outdir, filter='.png', keep_these_cats=[1,55,56,57], labels=constants.fashionista_categories_augmented):
    masklist = [f for f in os.listdir(indir) if filter in f]
#    print('masks:'+str(masklist))
#    labels = constants.pascal_context_labels
    final_labels = ['','null','hair','skin','face']
    final_labels = [labels[ind] for ind in keep_these_cats]
    final_labels[:0] = [''] #prepend
    print('final labels:'+str(final_labels))
    for mask in masklist:
        fullname = os.path.join(indir,mask)
        print('name:'+mask+' full:'+fullname)
 #       show_mask_with_labels(fullname,labels)
        mask_img = cv2.imread(fullname)
        if len(mask_img.shape)==3:
            print('fixing multichan mask')
            mask_img = mask_img[:,:,0]
        new_mask = defenestrate_labels(mask_img,keep_these_cats)
        outname = os.path.join(outdir,mask)
        cv2.imwrite(outname,new_mask)
        print('outname:'+outname+', uniques '+str(np.unique(new_mask)))
  #      show_mask_with_labels(outname,final_labels)

def concatenate_labels(mask,kplist):
    matches = np.ones_like(mask)
    first = kplist[0]
    for i in range(0,len(kplist)):
        index = kplist[i]
        nv = np.multiply(mask == index,first)
        print(nv.shape)
        matches = np.add(matches,nv)
    return matches

def resize_and_crop_maintain_bb_on_dir(dir, output_width = 150, output_height = 200,use_visual_output=True):
    only_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]
    print('doing resize/crop in dir '+dir)
#    print(only_files)
    for a_file in only_files:
        print('file '+a_file)
        fullfile = os.path.join(dir,a_file)
        retval = resize_and_crop_maintain_bb(fullfile, output_width = 150, output_height = 200,use_visual_output=True,bb=None)

def show_mask_with_labels_dir(dir,labels,filter=None,original_images_dir=None,original_images_dir_alt=None,cut_the_crap=False,save_images=False,visual_output=False,webtool=False):
    '''

    :param dir:
    :param filter:  take only images with this substring in name
    :param labels: list of test labels for categories
    :param original_images_dir: dir of image (not labels)
    :param original_images_dir_alt: alternate dir of images (to deal with test/train directories)
    :param cut_the_crap: sort images to keepers and tossers
    :return:
    '''
    if filter:
        print('using filter:'+filter)
        files = [f for f in os.listdir(dir) if filter in f]
    else:
        files = [f for f in os.listdir(dir) ]
    print(str(len(files))+ ' files to process in '+dir)
    fullpaths = [os.path.join(dir,f) for f in files]
    totfrac = 0
    fraclist=[]
    n=0
    if original_images_dir:
        original_images = ['.'.join(f.split('.')[:-1])+'.jpg' for f in files]
#        if webtool:
#            original_images = [f.replace('_pixv2','').replace('_webtool','') for f in files]
#        original_images = [f.split('.')[-2]+'.jpg' for f in files]
        original_fullpaths = [os.path.join(original_images_dir,f) for f in original_images]
        if original_images_dir_alt:
            original_altfullpaths = [os.path.join(original_images_dir_alt,f) for f in original_images]
        for x in range(0,len(files)):
            if os.path.exists(original_fullpaths[x]):
                 show_mask_with_labels(fullpaths[x],labels,original_image=original_fullpaths[x],cut_the_crap=cut_the_crap,save_images=save_images,visual_output=visual_output)
#                if frac is not None:
#                    fraclist.append(frac)
#                    totfrac = totfrac + frac
#                    n=n+1
            elif original_images_dir_alt and os.path.exists(original_altfullpaths[x]):
                show_mask_with_labels(fullpaths[x],labels,original_image=original_altfullpaths[x],cut_the_crap=cut_the_crap,save_images=save_images,visual_output=visual_output)
 #               if frac is not None:
 #                   fraclist.append(frac)
 ##                   totfrac = totfrac + frac
   #                 n=n+1
            else:
                logging.warning(' does not exist:'+original_fullpaths[x])
                continue

    else:
        for f in fullpaths:
            show_mask_with_labels(f,labels,cut_the_crap=cut_the_crap,save_images=save_images,visual_output=visual_output)
#            if frac is not None:
#                fraclist.append(frac)
#                totfrac = totfrac + frac
#                n=n+1
#    print('avg frac of image w nonzero pixels:'+str(totfrac/n))
    hist, bins = np.histogram(fraclist, bins=30)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width,label='nonzero pixelcount')
    plt.show()
    plt.legend()
    plt.savefig('outhist.jpg')
#    print('fraction histogram:'+str(np.histogram(fraclist,bins=20)))

def show_mask_with_labels(mask_filename_or_img_array,labels,original_image=None,cut_the_crap=False,save_images=False,visual_output=False,resize=None,mask2=None,overlay=None,savename=None):
    '''
    split this into one function that takes mask and gives img with labels possibly with overlay, returns arr
    and another func that takes 2 images and puts side by side
    todo : use overlay cv2.addWeighted(overlay, alpha, output, 1 - alpha,
		0, output)
		http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    colormap = cv2.COLORMAP_JET
    if isinstance(mask_filename_or_img_array, basestring):
        img_arr = Utils.get_cv2_img_array(mask_filename_or_img_array,cv2.IMREAD_GRAYSCALE)
        mask_filename=mask_filename_or_img_array
    elif type(mask_filename_or_img_array) == np.ndarray:
        img_arr = mask_filename_or_img_array
        mask_filename='./output.jpg'
        if original_image is not None and isinstance(original_image,basestring):
            mask_filename = original_image
    else:
        logging.warning('got something other than a filename (string) or img array')
        return
    if img_arr is None:
        logging.warning('img_arr is None')
        return
    logging.debug('img size:'+str(img_arr.shape))
    if len(img_arr.shape) != 2:
        logging.warning('got a multichannel image, using chan 0')
        img_arr = img_arr[:,:,0]
    histo = np.histogram(img_arr,bins=len(labels)-1)
#    print('hist'+str(histo[0])) #
    h,w = img_arr.shape[0:2]
    n_nonzero = np.count_nonzero(img_arr)
    n_tot = h*w
    frac = float(n_nonzero)/n_tot
    uniques = np.unique(img_arr)
    logging.debug('show_mask_with_labels:number of unique mask values:'+str(len(uniques))+' frac nonzero:'+str(frac) +' hxw:'+str(h)+','+str(w))
    if len(uniques)>len(labels):
        logging.warning('number of unique mask values {} > number of labels {}!!!'.format(len(uniques),len(labels)))
        return
    # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_array)
    maxVal = len(labels)
    max_huelevel = 160.0
    satlevel = 255
    vallevel = 255
    scaled = np.uint8(np.multiply(img_arr, max_huelevel / maxVal))
#        dest = cv2.applyColorMap(scaled,colormap)
    dest = np.zeros([h,w,3])
    dest[:,:,0] = scaled  #hue
    dest[:,:,1] = satlevel   #saturation
    dest[:,:,2] = vallevel   #value
 #   print('type:'+str(type(dest)))
    dest = dest.astype(np.uint8)
    dest = cv2.cvtColor(dest,cv2.COLOR_HSV2BGR)

    bar_height = int(float(h)/len(uniques))
    bar_width = 170
    colorbar = np.zeros([h,bar_width])
    i = 0
    logging.debug('len labels:'+str(len(labels)))
    logging.debug('unique label val:'+str(uniques))
    for unique in uniques:
        if unique > len(labels):
            logging.warning('pixel value '+str(unique)+' out of label range (1)')
            continue
        colorbar[i*bar_height:i*bar_height+bar_height,:] = unique

#        cv2.putText(colorbar,labels[unique],(5,i*bar_height+bar_height/2-10),cv2.FONT_HERSHEY_PLAIN,1,[i*255/len(uniques),i*255/len(uniques),100],thickness=2)
#        cv2.putText(colorbar,labels[unique],(5,i*bar_height+bar_height/2-5),cv2.FONT_HERSHEY_PLAIN,1,[0,10,50],thickness=2)
        i=i+1

    scaled_colorbar = np.uint8(np.multiply(colorbar, max_huelevel / maxVal))
    h_colorbar,w_colorbar = scaled_colorbar.shape[0:2]
    dest_colorbar = np.zeros([h_colorbar,w_colorbar,3])
    dest_colorbar[:,:,0] = scaled_colorbar  #hue
    dest_colorbar[:,:,1] = satlevel   #saturation
    dest_colorbar[:,:,2] = vallevel  #value
    dest_colorbar = dest_colorbar.astype(np.uint8)
    dest_colorbar = cv2.cvtColor(dest_colorbar,cv2.COLOR_HSV2BGR)
 #   print('size of colrbar:'+str(dest_colorbar.shape))
 #have to do labels here to get black
    i = 0
    totpixels = h*w
    for unique in uniques:
        if unique >= len(labels):
            logging.warning('pixel value '+str(unique)+' out of label range (2)')
            continue
        pixelcount = len(img_arr[img_arr==unique])
        try:
            logging.debug('unique:'+str(unique)+':'+labels[unique]+' pixcount:'+str(pixelcount)+' fraction'+str(float(pixelcount)/totpixels))
            frac_string='{:.4f}'.format(float(pixelcount)/totpixels)
            text_string = str(unique)+' '+labels[unique]+' '+str(frac_string)
            cv2.putText(dest_colorbar,text_string,(5,int(i*bar_height+float(bar_height)/2+5)),cv2.FONT_HERSHEY_PLAIN,0.7,[0,10,50],thickness=1)
        except:
#            logging.warning('some problem in labelling')
            print("Unexpected error:"+ str(sys.exc_info()[0]))
            print('index {} len labels {}'.format(unique,len(labels)))
        i=i+1 #

    #dest_colorbar = cv2.applyColorMap(scaled_colorbar, colormap)
    combined = np.zeros([h,w+w_colorbar,3],dtype=np.uint8)
    if mask2:
        combined = np.zeros([h,w+w_colorbar,3],dtype=np.uint8)
        mask2_arr = Utils.get_cv2_img_array(mask2,cv2.IMREAD_GRAYSCALE)
    combined[:,0:w_colorbar]=dest_colorbar
    combined[:,w_colorbar:w_colorbar+w]=dest
    if original_image is not None:
        orig_arr = Utils.get_cv2_img_array(original_image)
#        orig_arr = cv2.imread(original_image)
        if orig_arr is not None:
            height, width = orig_arr.shape[:2]
            logging.debug('show_mask_with_labels:got original image:'+str(original_image)+' shape:'+str(orig_arr.shape))
            maxheight=600
            minheight=300
            desired_height=500
            if resize:  # or height < minheight:
    #        if (1):  # or height < minheight:
                desired_height=resize[0]
                logging.debug('(hxw {}x{}) resizing to {} by '.format(height,width,desired_height))
#                newheight=(height>maxheight)*maxheight   #+(height<minheight)*minheight
                newheight=desired_height
                factor = float(newheight)/height
                orig_arr = cv2.resize(orig_arr,(int(round(width*factor)),int(round(height*factor))))
#                print('factor {} newsize {}'.format(factor,orig_arr.shape) )

                colorbar_h,colorbar_w = dest_colorbar.shape[0:2]
                factor = float(newheight)/colorbar_h
                dest_colorbar = cv2.resize(dest_colorbar,(int(round(colorbar_w*factor)),int(round(colorbar_h*factor))))
#                print('cbarfactor {} newsize {}'.format(factor,dest_colorbar.shape) )

                dest_h,dest_w = dest.shape[0:2]
                factor = float(newheight)/dest_h
                dest = cv2.resize(dest,(int(round(dest_w*factor)),int(round(dest_h*factor))))
#                print('maskfactor {} newsize {}'.format(factor,dest.shape) )

        #    cv2.imshow('original',orig_arr)
            elif height != h or width != w:
                orig_arr = resize_keep_aspect(orig_arr,output_size=(h,w))
                logging.debug('size mismach bet. orig and mask - orig {}x{} mask {}x{}'.format(height,width,h,w))
            colorbar_h,colorbar_w = dest_colorbar.shape[0:2]
#            logging.debug('dest colorbar w {} h {} shape {}'.format(colorbar_w,colorbar_h,dest_colorbar.shape))
            dest_h,dest_w = dest.shape[0:2]
#            logging.debug('dest w {} h {} shape {}'.format(dest_w,dest_h,dest.shape))
            orig_h,orig_w = orig_arr.shape[0:2]
            logging.debug('orig w {} h {} dest {}x{}'.format(orig_w,orig_h,dest_w,dest_h))
#            print('colobar size {} masksize {} imsize {}'.format(dest_colorbar.shape,dest.shape,orig_arr.shape))
            combined = np.zeros([dest_h,dest_w+orig_w+colorbar_w,3],dtype=np.uint8)
            logging.debug('show_mask_with_labels:combined shape:'+str(combined.shape))
            combined[:,0:colorbar_w]=dest_colorbar
            combined[:,colorbar_w:colorbar_w+dest_w]=dest
            if overlay:
                logging.debug('show_mask_with_labels:doing overlay')
                orig_arr = cv2.addWeighted(orig_arr, overlay, img_arr, 1 - overlay,0)
            combined[:,colorbar_w+dest_w:]=orig_arr
 #ValueError: could not broadcast input array from shape (572,940,3) into shape (256,940,3)

            combined_h,combined_w = combined.shape[0:2]
            logging.debug('show_mask_with_labels:comb w {} h {} shape {}'.format(combined_w,combined_h,combined.shape))
#            if combined_h<minheight:
#                factor = float(minheight)/combined_h
#                combined = cv2.resize(combined,(int(round(combined_w*factor)),minheight))
        else:
            logging.warning('show_mask_with_labels could not get image '+original_image)
 #   cv2.imshow('map',dest)
 #   cv2.imshow('colorbar',dest_colorbar)
    relative_name = os.path.basename(mask_filename)
    if visual_output:
        cv2.imshow(relative_name,combined)
        k = cv2.waitKey(0)
    if save_images:
        if savename is None:
            savename = mask_filename[:-4]+'_legend.jpg'
        logging.info('show_mask_with_labels is saving labelled img to '+savename)
        cv2.imwrite(savename,combined)

    #todo move this to a separate function i dont think theres any reason its here
    if cut_the_crap:  #move selected to dir_removed, move rest to dir_kept
        print('(d)elete (c)lose anything else keeps')
        indir = os.path.dirname(mask_filename)
        parentdir = os.path.abspath(os.path.join(indir, os.pardir))
        curdir = os.path.split(indir)[1]
        print('in {} parent {} cur {}'.format(indir,parentdir,curdir))
        if k == ord('d'):
            newdir = curdir+'_removed'
            dest_dir = os.path.join(parentdir,newdir)
            Utils.ensure_dir(dest_dir)
            print('REMOVING moving {} to {}'.format(mask_filename,dest_dir))
            shutil.move(mask_filename,dest_dir)

        elif k == ord('c'):
            newdir = curdir+'_needwork'
            dest_dir = os.path.join(parentdir,newdir)
            Utils.ensure_dir(dest_dir)
            print('CLOSE so moving {} to {}'.format(mask_filename,dest_dir))
            shutil.move(mask_filename,dest_dir)

        else:
            newdir = curdir+'_kept'
            dest_dir = os.path.join(parentdir,newdir)
            Utils.ensure_dir(dest_dir)
            print('KEEPING moving {} to {}'.format(mask_filename,dest_dir))
            shutil.move(mask_filename,dest_dir)

    cv2.destroyAllWindows()
    print('finished show_mask_with-labels')
    print('caller name:'+str( inspect.stack()[1][3]))
    return combined,frac
#        return dest
def resize_dir(dir,out_dir,factor=4,filter='.jpg'):
    imfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f)) and filter in f]
    for f in imfiles:
        infile = os.path.join(dir,f)
        img_arr = cv2.imread(infile)
        if filter == '.png' or filter=='.bmp' or filter == 'png' or filter == 'bmp':  #png mask is read as x*y*3 , prob. bmp too
            img_arr = img_arr[:,:,0]
        h, w = img_arr.shape[0:2]
        new_h = int(h/factor)
        new_w = int(w/factor)
        output_arr = cv2.resize(img_arr,(new_w,new_h))
        actualh,actualw = output_arr.shape[0:2]
        outfile = os.path.join(out_dir,f)
        cv2.imwrite(outfile,output_arr)
        print('orig w,h {},{} new {},{} '.format(w,h,actualw,actualh))
        print('infile {} outfile {}'.format(infile,outfile))

def nms_detections(dets, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    overlap: float
        minimum overlap ratio (0.3 default)

    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    ind = np.argsort(dets[:, 4])

    w = x2 - x1
    h = y2 - y1
    area = (w * h).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        xx1 = np.maximum(x1[i], x1[ind])
        yy1 = np.maximum(y1[i], y1[ind])
        xx2 = np.minimum(x2[i], x2[ind])
        yy2 = np.minimum(y2[i], y2[ind])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)

        wh = w * h
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :]

def img_dir_to_html(img_dir,filter='.jpg',htmlname=None):
    imglist = [i for i in os.listdir(img_dir) if filter in i]
    line_no=0
    lines=[]

    if htmlname is None:
        parentdir = os.path.abspath(os.path.join(img_dir, os.pardir))
        htmlname=parentdir+'.html'
        htmlname=img_dir.replace('/','_')+'.html'
        htmlname=img_dir.replace('/','')+'.html'
    with open(htmlname,'w') as f:
        lines.append('<HTML><HEAD><TITLE>results '+img_dir+' </TITLE></HEAD>\n')
        for img in imglist:
            f.write('<br>\n')
            link = '"'+os.path.join(img_dir,img)+'"'
            f.write('<img src='+link+'>')
            #f.write('<a href='+link+'>'+img+'</a>\n')
        f.write('</HTML>\n')
        f.close()

def do_for_all_files_in_dir(some_function,dir,filter='.jpg',**kwargs):
    '''
    why didnt i do this a year ago
    applies a function onto a dir of jpgs
    '''
    print(kwargs)
    print(**kwargs)
    files = [os.path.join(dir,f) for f in os.listdir(dir) if filter in f]
    for f in files:
        some_function(f,**kwargs)

def clothe_lots(clothing_dir,mannequin_dir,type='fullbody',n=10000,filter='gc'):
    clothes_files = [os.path.join(clothing_dir,f) for f in os.listdir(clothing_dir) if filter in f]
    mannequin_files = [os.path.join(mannequin_dir,f) for f in os.listdir(mannequin_dir)]
    print('{} clothes and {} mannequins'.format(len(clothes_files),len(mannequin_files)))
    n_done=0
    while(n_done<n):
        c=random.choice(clothes_files)
        m=random.choice(mannequin_files)
        print('{} is trying on {} n={}'.format(m,c,n_done))
        clothe_the_naked(c,m,type=type,filter=filter)
        n_done+=1
    # for c in clothes_files:
    #     for m in mannequin_files:
    #         print('{} is trying on {}'.format(m,c))
    #         clothe_the_naked(c,m,type=type)

def clothe_the_naked(clothing_img, mannequin_img,type='fullbody',max_rot=6,save = True,interactive=True,savedir='clothed',filter=filter):
    Utils.ensure_dir(savedir)
    f = background_removal.find_face_dlib_with_scores(mannequin_img)
    print(f)
    img_mannequin = Utils.get_cv2_img_array(mannequin_img)
    img_clothing = Utils.get_cv2_img_array(clothing_img)
    center = (img_mannequin.shape[1]/2,img_mannequin.shape[0]/2)
    angle = max_rot*np.random.randn(max_rot)[0]
    r = cv2.getRotationMatrix2D(center,angle,scale=1)
  #  print(r)
    clothing_rotated = cv2.warpAffine(img_clothing,r,(img_mannequin.shape[0],img_mannequin.shape[1]))#     cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)
  #  print('angle {}'.format(angle))
    if f['are_faces']:
        faces = f['faces']
        for face in faces:
            print(face)
            cv2.rectangle(img_mannequin,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,100,0),thickness=3)
 #   cv2.imshow('mannequin',img_mannequin)
    full_size=(256,256)
    reduction_factor = 0.7
    if type == 'fullbody':
        reduction_factor = 0.8
    clothes_size = (int(full_size[0]*reduction_factor),int(full_size[1]*reduction_factor))
    mannequin_resized = resize_keep_aspect(mannequin_img,output_size=full_size)
 #   print('clothes size:{}'.format(clothes_size))
    clothes_resized = resize_keep_aspect(clothing_rotated,output_size = clothes_size)

#    cv2.imshow('orig m',img_mannequin)
#    cv2.imshow('clothing rotated',clothing_rotated)
#    cv2.imshow('mannequin_resized',mannequin_resized)
#    cv2.imshow('clothes_resized',clothes_resized)
#    k = cv2.waitKey(0)
#    cv2.destroyAllWindows()
    if filter:   # these ones have already been interactively gc'd so no need to gc
        p0 = clothes_resized[:,:,0]
        p1 = clothes_resized[:,:,1]
        p2 = clothes_resized[:,:,2]
        nonzero = np.where((p0!=0)+(p1!=0)+(p2!=0),255,0)
        print('size of nonzero {} type {}'.format(nonzero.shape,nonzero.dtype))
        nonzero = np.array(nonzero,dtype=np.uint8)
        print('size of nonzero {} type {}'.format(nonzero.shape,nonzero.dtype))
#        cv2.imshow('themask',nonzero)
#        cv2.waitKey(0)
        #mask2 = np.where((mask == cv2.GC_FGD) + (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)  #return all fg and prob. fg
        result = overlay(nonzero, clothes_resized,mannequin_resized)

    else:
        result = gc_then_overlay(clothes_resized,mannequin_resized)
    if result is None:
        pass
    elif save:
        if isinstance(mannequin_img,basestring):
            mannequin_name=os.path.basename(mannequin_img)
        else:
            mannequin_name='body'+''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
        if isinstance(clothing_img,basestring):
            clothing_name=os.path.basename(clothing_img)
        else:
            clothing_name='clothing'+''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
        name = mannequin_name.replace('.jpg','')+clothing_name.replace('gc.png','').replace('.jpg','').replace('.png','')+'.jpg'
        name = os.path.join(savedir,name)
        print('saving image to {}'.format(name))
        cv2.imwrite(name,result)
    else:
        print('not saving')

    # elif interactive:
    #     k=raw_input('s or return to save...')
    #     if k == 's' or k== '':
    cv2.destroyAllWindows()

def gc_then_overlay(im1,im2, position=None,save=True,visual_output=True):
    im1 = get_cv2_img_array(im1)
    im2 = get_cv2_img_array(im2)
    if im1.shape[0]>im2.shape[0] or im1.shape[1]>im2.shape[1]:
        print('overlay larger than image im1 {} im2 {}'.format(im1.shape,im2.shape))
        return
    if position == None:
        position = (im2.shape[0]/2,im2.shape[1]/2)
    mask_y = (im2.shape[0]-im1.shape[0])/2
    mask_x = (im2.shape[1]-im1.shape[1])/2

    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    mask = np.zeros(im1.shape[:2], dtype=np.uint8)

    #TODO - maybe find something better than median as the threshold

    # x0, x1, y0, y1 = []
    # mask[y0:y1, x0:x1] = 0
    # print('BG'+str(rectangle))

# cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD, or

    #prob. backgnd - entire image
    h,w = im1.shape[0:2]
    x0, x1, y0, y1 = [0,w,0,h]
    mask[y0:y1, x0:x1] = cv2.GC_PR_BGD
#    print('PBG x0 {} x1 {} y0 {} y1 {} '.format(x0,x1,y0,y1))

    #prob. fgnd - center rectangle
    bb_percent_w = 0.5  #percent of image center to use as bb
    bb_percent_h = 0.8  #percent of image center to use as bb
    w = int(im1.shape[1]*bb_percent_w)
    h = int(im1.shape[0]*bb_percent_h)
    x = int((im1.shape[1]-w)/2)
    y = int((im1.shape[0]-h)/2)
    x0, x1, y0, y1 = [x,x+w,y,y+h]
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    print('PFG x0 {} x1 {} y0 {} y1 {} '.format(x0,x1,y0,y1))

    #prob. fgnd - center rectangle
    bb_percent = 0.1  #percent of image center to use as bb
    w = int(im1.shape[1]*bb_percent)
    h = int(im1.shape[0]*bb_percent)
    x = int((im1.shape[1]-w)/2)
    y = int((im1.shape[0]-h)/2)
    x0, x1, y0, y1 = [x,x+w,y,y+h]
    mask[y0:y1, x0:x1] = cv2.GC_FGD
 #   print('FG x0 {} x1 {} y0 {} y1 {} '.format(x0,x1,y0,y1))


    try:
        #TODO - try more than 1 grabcut call in itr
        itr = 2
        cv2.grabCut(im1, mask, None, bgdmodel, fgdmodel, itr, cv2.GC_INIT_WITH_MASK) #im, mask, rect, bgmodel, fgmoel, iterations
    except:
        print('grabcut exception')
        return None
    mask2 = np.where((mask == cv2.GC_FGD) + (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)  #return all fg and prob. fg
#    mask = background_removal.get_fg_mask(im1,bounding_box=bb)
#    print('got mask shape {} uniques {} '.format(mask.shape,np.unique(mask)))
#    cv2.imshow('mask_b4gc',mask)
#    cv2.imshow('mask_aftergc',mask2)
#    cv2.waitKey(0)

    overlaid = overlay(mask2, im1,im2)
    return overlaid

def overlay(im1_mask,im1, bgnd_img,position=None,rotation=0,scale=1,save=True,visual_output=True):
    bgnd_img = Utils.get_cv2_img_array(bgnd_img)
    w,h = im1.shape[0:2]
    if im1_mask.shape[0]>bgnd_img.shape[0] or im1_mask.shape[1]>bgnd_img.shape[1]:
        print('overlay larger than image im1 {} im2 {}'.format(im1_mask.shape,bgnd_img.shape))
        return
    if position == None:
        im2,contours,hierarchy = cv2.findContours(im1_mask, 1, 2)
#        cv2.imshow('mask1',im1_mask)
#        cv2.waitKey(0)
        cnt = contours[0]
        M = cv2.moments(cnt)
     #   print('contour moments:'+str(M))
      #  From this moments, you can extract useful data like area, centroid etc. Centroid is given by the relations, Cx=M10M00 and Cy=M01M00. This can be done as follows:
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print('cx {} cy {}'.format(cx,cy))
        except:
            print('prob division by zero, m00={}'.format(M['m00']))
            cx = im1_mask.shape[0]/2
            cy = im1_mask.shape[1]/2
#        cv2.circle(im1_mask,(cx,cy),20,(255,100,50),thickness=5)
#        cv2.rectangle(img_arr,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color=(255,255,0),thickness=2)
  #      cv2.imshow('mask1',im1_mask)
  #      cv2.waitKey(0)
        dx = im1_mask.shape[0]/2-cx
        dy = im1_mask.shape[1]/2-cy
        position = (dx,dy)
        print('cx {} cy {} dx {} dy {}'.format(cx,cy,dx,dy))

    print('shifting by {}'.format(position))
    translation_matrix = np.float32([ [1,0,position[1]], [0,1,position[0]]] )
    im1_mask = cv2.warpAffine(im1_mask, translation_matrix, (w, h)) # cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)
    im1 = cv2.warpAffine(im1, translation_matrix, (w, h))   #cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)
    if scale != 1:
        print('im1_mask {} im1 {} before resize'.format(im1_mask.shape,im1.shape))
        h,w = im1.shape[0:2]
        dsize = (int(w*scale),int(h*scale))
        im1_mask = cv2.resize(im1_mask,dsize)
        im1 = cv2.resize(im1,dsize)
        print('im1_mask {} im1 {} after resize'.format(im1_mask.shape,im1.shape))
        if scale>1: #crop extra
            extra = (dsize[0]-h,dsize[1]-w)
            starty=extra[0]/2
            endy = extra[0]/2+h
            startx=extra[1]/2
            endx = extra[1]/2+w
            print('sy {} endy {} sx {} edx {}'.format(starty,endy,startx,endx))
            im1 = im1[starty:endy,startx:endx,:]
            im1_mask=im1_mask[starty:endy,startx:endx]
            print('im1_mask {} im1 {} after crop'.format(im1_mask.shape,im1.shape))
        else: #add missing
            extra = (h-dsize[0],w-dsize[1])
            print('extra {} h {} w {} dsize {} e0 {} e1 {}'.format(extra,h,w,dsize,extra[0],extra[1]))
            starty=extra[0]/2
            endy = extra[0]/2+dsize[0]
            startx=extra[1]/2
            endx = extra[1]/2+dsize[1]
            print('sy {} endy {} sx {} edx {}'.format(starty,endy,startx,endx))
            im1_dest = np.zeros((h,w,3))
            im1_mask_dest = np.zeros((h,w))
            im1_dest[starty:endy,startx:endx,:]= im1
            im1_mask_dest[starty:endy,startx:endx]=im1_mask
            print('im1_mask {} im1 {} after padding'.format(im1_mask.shape,im1.shape))

    if rotation != 0:
        center = (w/2,h/2)
        r = cv2.getRotationMatrix2D(center,rotation,scale=1)
        im1_mask = cv2.warpAffine(im1_mask, r, (w, h)) # cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)
        im1 = cv2.warpAffine(im1, r, (w, h))   #cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)


    mask_y = (bgnd_img.shape[0]-im1_mask.shape[0])/2
    mask_x = (bgnd_img.shape[1]-im1_mask.shape[1])/2

    final_canvas = np.zeros_like(bgnd_img)
    mask_height = im1_mask.shape[0]
    mask_width = im1_mask.shape[1]
    mask_on_canvas = np.zeros_like(bgnd_img)
    mask_on_canvas[mask_y:mask_y+mask_height,mask_x:mask_x+mask_width,0] = im1[:,:,0]
    mask_on_canvas[mask_y:mask_y+mask_height,mask_x:mask_x+mask_width,1] = im1[:,:,1]
    mask_on_canvas[mask_y:mask_y+mask_height,mask_x:mask_x+mask_width,2] = im1[:,:,2]

    print('im1 {} bgndd {} final canvas {} maskh {} maskw {}'.format(im1_mask.shape,bgnd_img.shape,final_canvas.shape,mask_height,mask_width))
    final_canvas[mask_y:mask_y+mask_height,mask_x:mask_x+mask_width,0] = im1_mask
    final_canvas[mask_y:mask_y+mask_height,mask_x:mask_x+mask_width,1] = im1_mask
    final_canvas[mask_y:mask_y+mask_height,mask_x:mask_x+mask_width,2] = im1_mask
    masked_1 = np.where(final_canvas!=0,mask_on_canvas,bgnd_img)
    if visual_output:
#        cv2.imshow('mask1',im1_mask)
 #       cv2.imshow('mask_on_canvas',mask_on_canvas)
 #       cv2.imshow('final',final_canvas)
 #       cv2.imshow('bgnd',bgnd_img)
        cv2.imshow('masked_1',masked_1)
        print('use arrow keys to translate:awds rotate:er scale:o-,p+ (q)uit, return to save')
        k = cv2.waitKey(0)

        #shift mask interactively
        print('pressed value:'+str(k))
        shift = 5 #pixels to translate each time
        if k == 37 or k ==  ord('a'): #left
            return(overlay(im1_mask,im1,bgnd_img,position=(0,-shift)))
        elif k == 38 or k ==  ord('w'): #up
            return(overlay(im1_mask,im1,bgnd_img,position=(-shift,0)))
        elif k == 39 or k ==  ord('d'): #right
            return(overlay(im1_mask,im1,bgnd_img,position=(0,+shift)))
        elif k == 40 or k ==  ord('s'): #down
            return(overlay(im1_mask,im1,bgnd_img,position=(shift,0)))
        elif k == ord('+') or k==ord('p'): #enlargen
            return(overlay(im1_mask,im1,bgnd_img,scale=1.05))
        elif k == ord('-') or k==ord('o'): #smallen
            return(overlay(im1_mask,im1,bgnd_img,scale=.95))
        elif k == ord('e'): #rot-
            return(overlay(im1_mask,im1,bgnd_img,rotation=-shift))
        elif k == ord('r'): #rot+
            return(overlay(im1_mask,im1,bgnd_img,rotation=shift))
        elif k == ord('q'): #quit
            return

    return masked_1
#    overlaid = np.where(mask_3channels>0,im1,im2)

def get_fg_mask(image, bounding_box=None):
    rect = (0, 0, image.shape[1]-1, image.shape[0]-1)
    bgdmodel = np.zeros((1, 65), np.float64)  # what is this wierd size about? (jr)
    fgdmodel = np.zeros((1, 65), np.float64)

    # bounding box was sent from a human - grabcut with bounding box mask
    if Utils.legal_bounding_box(bounding_box):
        if Utils.all_inclusive_bounding_box(image, bounding_box):  # bb is nearly the whole image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
        else:
            mask = bb_mask(image, bounding_box)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    # grabcut on the whole image, with/without face
    else:
        faces_dict = find_face_cascade(image)
        # if len(faces_dict['faces']) > 0:  # grabcut with mask
        #     try:
        #         rectangles = body_estimation(image, faces_dict['faces'][0])
        #         mask = create_mask_for_gc(rectangles, image)
        #     except:
        #         mask = create_mask_for_gc(image)
        #
        # else:  # grabcut with arbitrary rect
        mask = create_arbitrary(image)
        cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype(np.uint8)
    return mask2

def smallify_and_implant(arr_url_or_file,reduction_percent=30,background_image=None,bb=None,fade_in=True):
    '''
    WIP - finish this to augment yolo stuff - and call it from augment_images , checking size of largest object
    and smallifying accordingly. so we have to keep track of bb's too and return those smallified in same way
    :param arr_url_or_file:
    :param reduction_percent:
    :param background_image:
    :return:
    '''
    img_arr = Utils.get_cv2_img_array(arr_url_or_file)
    orig_h,orig_w = img_arr.shape[0:2]
    if background_image is not None:
        new_arr = resize_keep_aspect(background_image,output_size=(orig_h,orig_w))
    else:
        new_arr = np.zeros_like(img_arr)
    dsize=(orig_w*(1-reduction_percent),orig_h*(1-reduction_percent))# #make sure resize wants width,height not height,width
    reduced = cv2.resize(img_arr,dsize)
    x_wiggleroom = orig_w - dsize[0]
    y_wiggleroom = orig_h - dsize[1]

def dominant_colors(img_arr,n_components=2):
    '''
    :param img_arr: this will generally be a subimage (orig image cropped to a bb)
    :return:
    '''
    dom_color = None
    if img_arr is None:
        print('got non arr in dominant_colors')
        return None

    hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
    if hsv is None:
        print('some prob with hsv')
        return None

    try:
        avg_hue = np.mean(hsv[:,:,0])
        avg_sat = np.mean(hsv[:,:,1])
        avg_val = np.mean(hsv[:,:,2])
        stdev_hue = np.std(hsv[:,:,0])
        stdev_sat = np.std(hsv[:,:,1])
        stdev_val = np.std(hsv[:,:,2])
    #take care of large std for red (which wraps around from 180 to 0
        if stdev_hue>60:
            print('correcting hue modulo, orig mean {} std {}'.format(avg_hue,stdev_hue))
            hue=hsv[:,:,0]
            mask=hue>90
            hue=hue-mask*180
            avg_hue = np.mean(hue)
            stdev_hue = np.std(hue)
            print('corrected hue modulo, new mean {} std {}'.format(avg_hue,stdev_hue))
    except:
        print('problem calculating sat or val')

    print('avg hue {} std {} avg sat {} std {} avg val {} std {}'.format(avg_hue,stdev_hue,avg_sat,stdev_sat,avg_val,stdev_val))
    min_sat_for_color = 0.3*255  #102
    min_val_for_color=0.3*255 #76
    max_std_for_color=70
    max_val_for_black=0.35*255 #89
    min_val_for_white=0.8*255 #204
    max_sat_for_white=0.15*255 #38
    max_sat_for_gray=0.1*255
    max_val_for_gray=0.8*255
    min_val_for_gray=0.3*255

    if avg_sat > min_sat_for_color and avg_val > min_val_for_color and stdev_hue<max_std_for_color: #color in visible range
    #    print('got visible color')
        colors = ['red','orange','yellow','green','aqua','blue','purple','pink','red']
 #       range_edges=[20,45,70,140,180,260,290,291,340] #for range 0-360
        range_edges=[13,22,35,75,90,130,145,170,180]
        i=0
        while(avg_hue>range_edges[i]):
            i=i+1
       # i=i-1
     #   print('range edge '+str(i)+' color '+colors[i])
        dom_color = colors[i]
    elif avg_val < max_val_for_black:
      #  print('got black')
        dom_color = 'black'
    elif avg_val>min_val_for_white and avg_sat<max_sat_for_white:
       # print('got white')
        dom_color = 'white'
    elif avg_val<max_val_for_gray and avg_val>min_val_for_gray and avg_sat<max_sat_for_gray:
        dom_color='gray'

    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    debug=False
    if(debug):
        chans = cv2.split(hsv)
        colors = ("b", "g", "r")
        plt.figure()
        plt.title("'Flattened' Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        features = []

        # loop over the image channels
        for (chan, color) in zip(chans, colors):
            # create a histogram for the current channel and
            # concatenate the resulting histograms for each
            # channel
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)

            # plot the histogram
            plt.plot(hist, color = color)
            plt.xlim([0, 256])
        blu_patch = mpatches.Patch(color='blue', label='Hue')
#        plt.legend(handles=[blu_patch])
        grn_patch = mpatches.Patch(color='green', label='Sat')
#        plt.legend(handles=[grn_patch])
        red_patch = mpatches.Patch(color='red', label='Val')
        plt.legend(handles=[red_patch,blu_patch,grn_patch])
        # here we are simply showing the dimensionality of the
        # flattened color histogram 256 bins for each channel
        # x 3 channels = 768 total values -- in practice, we would
        # normally not use 256 bins for each channel, a choice
        # between 32-96 bins are normally used, but this tends
        # to be application dependent
        print("flattened feature vector size: %d" % (np.array(features).flatten().shape))
        plt.show()
    print('dominant color:'+str(dom_color))
    return dom_color

def test_dominant_colors():
    images = ['white.jpg','black.jpg','pink.jpg','red.jpg','orange.jpg','yellow.jpg','green.jpg','blue.jpg','lightblue.jpg','purple.jpg',
              'orange.jpg','grey.jpg','turqoise.jpg']
    for im in images:
        path = os.path.join('/home/jeremy/projects/core/images',im)
        img_arr = cv2.imread(path)
        col = dominant_colors(img_arr,n_components=2)
        print('file:{} color {}'.format(path,col))

def browse_images(dir,filter='.jpeg'):
    files = [os.path.join(dir,f) for f in os.listdir(dir) if filter in f]
    for f in files:
        img_arr = cv2.imread(f)
        cv2.imshow('img',img_arr)
        cv2.waitKey(0)


def one_person_per_image(image,save_dir='multiple_people',visual_output=False):
    if isinstance(image,basestring):
#        imgname = image.replace('https://','').replace('http://','').replace('/','_') #conver url to name
        imgname = image
    else:
        imgname = 'test.jpg'
    img_arr = Utils.get_cv2_img_array(image)
    faces = background_removal.find_face_dlib_with_scores(img_arr)
    print(faces)
    if 'scores' in faces and 'faces' in faces:
        for score,bbox in zip(faces['scores'],faces['faces']):
            print('score {} bbox {}'.format(score,bbox))
            cv2.rectangle(img_arr,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color=(255,255,0),thickness=2)
        if len(faces['scores'])>1:
            multiples_dir = os.path.join(os.path.dirname(image),save_dir)
            Utils.ensure_dir(multiples_dir)
            savename = os.path.join(multiples_dir,os.path.basename(imgname))
            print('more than one face found, moving {} to {}'.format(image,savename))
            mvcmd = 'mv '+imgname+' '+savename
            subprocess.call(mvcmd,shell=True)
        if visual_output:
            cv2.imshow('image',img_arr)
            cv2.waitKey(100)

def x1y1x2y2_to_xywh(bb):
    assert bb[2]>bb[0],'bb not in format x1y1x2y2 {}'.format(bb)
    assert bb[3]>bb[1],'bb not in format x1y1x2y2 {}'.format(bb)
    return [bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]]

def xywh_to_x1y1x2y2(bb):
    return [bb[0],bb[1],bb[2]+bb[0],bb[3]+bb[1]]

def xywh_to_yolo(bb_xywh,dims_hxw,correct_out_of_bounds=True):
    '''
    output : for yolo - https://pjreddie.com/darknet/yolo/
    Darknet wants a .txt file for each image with a line for each ground truth object in the image that looks like:
    <object-class> <x> <y> <width> <height>
    where those are percentages and x,y are CENTER OF BB (also in percent)
    :param bb_xywh:
    :param image_dims size of image for this bb (needed since yolo wants bb's as percentages)
    :return:
    '''
    if correct_out_of_bounds:
        if bb_xywh[0] > dims_hxw[1]:
            bb_xywh[0] = dims_hxw[1]
            logging.warning('corrected y out of bounds')
        if bb_xywh[1] > dims_hxw[0]:
            bb_xywh[1] = dims_hxw[0]
            logging.warning('corrected x out of bounds!')
        if bb_xywh[0]+bb_xywh[2] > dims_hxw[1]:
            bb_xywh[2] = dims_hxw[1]-bb_xywh[0]
            logging.warning('corrected x+w > image width!!')
        if bb_xywh[1]+bb_xywh[3] > dims_hxw[0]:
            bb_xywh[3] = dims_hxw[0]-bb_xywh[1]
            logging.warning('corrected y+h > image height!!')

    x_center = bb_xywh[0]+(bb_xywh[2]/2.0)   #x1+w/2
    y_center = bb_xywh[1]+(bb_xywh[3]/2.0)    #y1+h/2
    x_p = float(x_center)/dims_hxw[1]    #center x as %
    y_p = float(y_center)/dims_hxw[0]   #center y as %
    w_p = float(bb_xywh[2])/dims_hxw[1] #width as %
    h_p = float(bb_xywh[3])/dims_hxw[0]  #height as %

    try:
        assert x_p<=1,'x > image width!!'
    except:
        logging.warning('x_p>1 bb {} out of bounds hw {}'.format(bb_xywh,dims_hxw))
    try:
        assert y_p<=1,'y > image height!!'
    except:
        logging.warning('y_p > 1 bb {} out of bounds hw {}'.format(bb_xywh,dims_hxw))
    try:
        assert bb_xywh[0]+bb_xywh[2]<=dims_hxw[1],'x+w > image width!!'
    except:
        logging.warning('x+width bb {} out of bounds hw {}'.format(bb_xywh,dims_hxw))
    try:
        assert bb_xywh[1]+bb_xywh[3]<=dims_hxw[0],'y+h > image height!!'
    except:
        logging.warning('y+height bb {} out of bounds hw {}'.format(bb_xywh,dims_hxw))

    return([x_p,y_p,w_p,h_p])

def x1x2y1y2_to_yolo(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def yolo_to_xywh(bb_yolo,image_dims_HxW):  #should change this to HxW and all callers, what was i thiinking
    '''
    output : for yolo - https://pjreddie.com/darknet/yolo/
    Darknet wants a .txt file for each image with a line for each ground truth object in the image that looks like:
    :param bb_yolo: x_center, y_center, w, h all as percentages of image width or height
    :param image_dims size of image for this bb (needed since yolo wants bb's as percentages)
    :return:
    '''

    x_center = float(bb_yolo[0])*image_dims_HxW[1]    #center x in pixels
    y_center = float(bb_yolo[1])*image_dims_HxW[0]   #center y pixels
    w = float(bb_yolo[2])*image_dims_HxW[1] #width pixels
    h = float(bb_yolo[3])*image_dims_HxW[0]  #height pixels
    x=x_center-w/2
    y=y_center-h/2
    logging.debug('in {} dims {} out(xywh) {} {} {} {}'.format(bb_yolo,image_dims_HxW,x,y,w,h))
    return([int(x),int(y),int(w),int(h)])

def bb_with_text(img_arr,bb_xywh,text,boxcolor = [50,255,50],text_bgnd_color=[255,255,80],box_thickness=1):
    text_color=[0,50,255]
    cv2.rectangle(img_arr,(bb_xywh[0],bb_xywh[1]),(bb_xywh[0]+bb_xywh[2],bb_xywh[1]+bb_xywh[3]),color=boxcolor,thickness=box_thickness)
    img_arr[bb_xywh[1]:bb_xywh[1]+20,bb_xywh[0]:bb_xywh[0]+bb_xywh[2]]=(img_arr[bb_xywh[1]:bb_xywh[1]+20,bb_xywh[0]:bb_xywh[0]+bb_xywh[2]]/2)+np.array(text_bgnd_color)/2
    cv2.putText(img_arr,text,(bb_xywh[0]+5,bb_xywh[1]+20),cv2.FONT_HERSHEY_PLAIN, 1, text_color)
    return img_arr

def count_values(mask,labels=None):
    image_size = mask.shape[0]*mask.shape[1]
    uniques = np.unique(mask)
    pixelcounts = {}
    for unique in uniques:
        pixelcount = len(mask[mask==unique])
        ratio = float(pixelcount)/image_size
        if labels is not None:
            print('class {} {} count {} ratio {}'.format(unique,labels[unique],pixelcount,ratio))
        else:
            print('class {} count {} ratio {}'.format(unique,pixelcount,ratio))
        pixelcounts[unique]=pixelcount
    return pixelcounts

def get_median_image(img_arr_list,visual_output=True):
    ''''
    given list of image arrs, produce median image useful for bg subtraction
    '''
    np_images = np.array(img_arr_list)
    print('np size:'+str(np_images.shape))
    median_image = np.median(np_images,axis=0) #get median pixel across images
    print('type:'+str(type(median_image)))
    median_image = np.array(median_image,dtype=np.uint8)
    print('median size:'+str(median_image.shape))
    if visual_output:
        cv2.imshow('median',median_image)
        k=cv2.waitKey(0)
    return median_image


def test_median_image():
    dir = '/home/jeremy/PycharmProjects/snooker/'
    files = [file for file in os.listdir(dir) if '.jpg' in file]
    files = sorted(files)
 #   build  image array
    img_arr_list =[]
    # for file in files:
    #     path = os.path.join(dir,file)
    #     img_arr = cv2.imread(path)
    #     img_arr_list.append(img_arr)
    #
    # med_img = get_median_image(img_arr_list)
    # cv2.imwrite(os.path.join(dir,'median.bmp'),med_img)

    med_img = cv2.imread(os.path.join(dir, 'median2.bmp'))
    med_eq = clahe_rgb(med_img)

    cv2.imshow('hi',med_img)
    cv2.imshow('eq',med_eq)
    cv2.waitKey(0)
    height, width, channels = med_img.shape

    outfile = os.path.join(dir, 'out.mp4')
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(outfile, fourcc, 20.0, (width, height))


    import time
    start = time.time()
    for file in files:
        path = os.path.join(dir, file)
        img_arr = cv2.imread(path)
        img_eq = clahe_rgb(img_arr)
        diff = cv2.subtract(img_eq, med_eq)
        cv2.imshow('diff',diff)
        cv2.waitKey(10)

        print('ok1')
        out.write(diff)  # Write out frame to video
        print('ok2')

    elapsed = time.time() - start
    print(
        'elapsed {} n {} tpi {} ipt {} '.format(elapsed, len(files), elapsed / len(files), float(len(files)) / elapsed))

def clahe_rgb(img_arr):
    #-----Converting image to LAB Color model-----------------------------------
    lab= cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB)
#    cv2.imshow("lab",lab)
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)
    #     #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
#    cv2.imshow('CLAHE output', cl)
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
 #   cv2.imshow('limg', limg)
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
  #  cv2.imshow('final', final)
    return final

if __name__ == "__main__":
    test_median_image()
    img=cv2.imread('../images/female1.jpg')
    resize_by_adding_border(img,output_size=(900,1000),visual_output=True)
#    test_or_training_textfile('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train',test_or_train='test')
 #   test_or_training_textfile('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train',test_or_train='train')
#    Utils.remove_duplicate_files('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1/')
#    resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,170],output_w=50,output_h=50)
 #   resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,400],output_w=50,output_h=50)
  #  resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,400],output_w=150,output_h=50)
   # resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,400],output_w=50,output_h=150)
    #resize_and_crop_image_using_bb('../images/female1.jpg',bb=[240,122,170,170],output_w=1000,output_h=100)
#    avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles = image_stats_from_dir_of_dirs(dir_of_dirs,filter='test')
 #   print('avg h {} avg w {} avgB {} avgG {} avgR {} nfiles {} in dir_of_dirs {}',avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles,dir_of_dirs)
#    dir_of_dirs = '/home/jr/core/classifier_stuff/caffe_nns/dataset'
#    raw_input('enter to continue')
  #  image_chooser_dir_of_dirs(dir_of_dirs,output_dir)
#    image_chooser(dir_of_dirs,output_dir)
#    crop_files_in_dir_of_dirs(dir_of_dirs,bb=None,output_w =150,output_h =200,use_visual_output=True)
#        dir = '/home/jeremy/projects/core/images'
#        resize_and_crop_maintain_bb_on_dir(dir, output_width = 448, output_height = 448,use_visual_output=True)

    if(0): #test mask to bbs
        # url = 'http://s-media-cache-ak0.pinimg.com/736x/fe/5d/f7/fe5df7e80093f674ecc79a9f30069a8a.jpg'
        # start=time.time()
        # retval = neurodoll_falcon_client.nd(url,get_combined_results=True)
        #
        # elapsed = time.time()-start
        # print('elapsed time in nd:'+str(elapsed))
        # if retval['success']:
        #     print('got nd')
        #     cv2.imwrite('/home/jeremy/projects/core/images/dress_mask_u21.png',retval['mask'])
        #     mask_to_rects(retval['mask'])
        # else:
        #     print('did not get good mask from ndfc')
        mask = cv2.imread('/home/jeremy/projects/core/images/dress_mask_u21.png')
        bbs = mask_to_rects(mask,visual_output=True)
        print('bbs:{}'.format(bbs))

    if(0) :  #test dominant colors
        dir = '/home/jeremy/Dropbox/tg/color_snatches'
        files = [os.path.join(dir,f) for f in os.listdir(dir)]
        for file in files:
            print('file '+file)
            im1=cv2.imread(file)
            cv2.imshow('im1',im1)
            cv2.waitKey(0)
            dominant_colors(im1)


#     dir = '/home/jeremy/tg/pd_output'
#     dir = '/root'
#     indir = '/home/jeremy/image_dbs/fashionista-v0.2.1'
#     outdir = '/home/jeremy/image_dbs/fashionista-v0.2.1/reduced_cats'
#
#     indir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_200x150'
#     outdir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_200x150/reduced_cats'
# #    defenestrate_directory(indir,outdir,filter='.png',keep_these_cats=[1,55,56,57],labels=constants.fashionista_categories_augmented)
#
#     if host == 'jr-ThinkPad-X1-Carbon' or host == 'jr':
#         dir_of_dirs = '/home/jeremy/tg/train_pairs_dresses'
#         output_dir = '/home/jeremy/tg/curated_train_pairs_dresses'
#         sourcedir = '/home/jeremy/projects/core/d1'
#         targetdir = '/home/jeremy/projects/core/d2'
#         infile =  '/home/jeremy/projects/core/images/female1.jpg'
#     else:
#         dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/dataset/cropped'
#         output_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/curated_dataset'
#
#  #   kill_the_missing(sourcedir, targetdir)
#
#     image_chooser('/data/jeremy/image_dbs/tg/google/pijamas - Google Search_files')
#
#     output_file = 'resized.jpg'
#     img_arr = cv2.imread(infile)
#     orig_h,orig_w = img_arr.shape[0:2]
#
#     resize_keep_aspect(infile, output_file=output_file, output_size = (600,400),use_visual_output=True)
#     undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)
#
#     resize_keep_aspect(infile, output_file=output_file, output_size = (600,401),use_visual_output=True)
#     undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)
#
#     resize_keep_aspect(infile, output_file=output_file, output_size = (600,399),use_visual_output=True)
#     undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)
#
#     resize_keep_aspect(infile, output_file=output_file, output_size = (400,600),use_visual_output=True)
#     undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)
#
#     resize_keep_aspect(infile, output_file=output_file, output_size = (400,601),use_visual_output=True)
#     undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)
#
#     resize_keep_aspect(infile, output_file=output_file, output_size = (400,599),use_visual_output=True)
#     undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)
#

#nonlinear xforms , stolen from:
#https://www.kaggle.com/bguberfain/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation/comments
'''
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
In [3]:
# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

# Load images
im = cv2.imread("../input/train/10_1.tif", -1)
im_mask = cv2.imread("../input/train/10_1_mask.tif", -1)

# Draw grid lines
draw_grid(im, 50)
draw_grid(im_mask, 50)

# Merge images into separete channels (shape will be (cols, rols, 2))
im_merge = np.concatenate((im[...,None], im_mask[...,None]), axis=2)
In [4]:
# First sample...

%matplotlib inline

# Apply transformation on image
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

# Split image and mask
im_t = im_merge_t[...,0]
im_mask_t = im_merge_t[...,1]

# Display result
plt.figure(figsize = (16,14))
plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')


# Second sample (heavyer transform)...

%matplotlib inline

# Apply transformation on image
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.09)

# Split image and mask
im_t = im_merge_t[...,0]
im_mask_t = im_merge_t[...,1]

# Display result
plt.figure(figsize = (16,14))
plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')

 managed to get about 4x improvement by using:

# include 4 standard deviations in the kernel (the default for ndimage.gaussian_filter)
# OpenCV also requires an odd size for the kernel hence the "| 1" part
blur_size = int(4*sigma) | 1
cv2.GaussianBlur(image, ksize=(blur_size, blur_size), sigmaX=sigma)
instead of ndimage.gaussian_filter(image, sigma)

and cv2.remap(image, dx, dy, interpolation=cv2.INTER_LINEAR) instead of ndimage.map_coordinates(image, (dx, dy), order=1)

    resize_keep_aspect(infile, output_file=None, output_size = (300,200),use_visual_output=True)
'''
