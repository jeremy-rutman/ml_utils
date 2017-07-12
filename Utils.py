from __future__ import print_function

import multiprocessing

__author__ = 'liorsabag'
import csv
import gzip
import json
import requests
from cv2 import imdecode, imwrite
import logging
import os
from requests import ConnectionError
import time
import numpy as np
from bson import objectid
import constants
import math
import cv2
import re
import string
import sys
import hashlib
logging.basicConfig(level=logging.WARNING)
USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'
db = constants.db


def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.

Note: this method may produce invalid filenames such as ``, `.` or `..`
When I use this method I prepend a date string like '2009_01_15_19_46_32_'
and append a file extension like '.txt', so I avoid the potential of using
an invalid filename.

"""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')  # I don't like spaces in filenames.
    return filename


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


class GZipCSVReader:
    def __init__(self, filename):
        self.gzfile = gzip.open(filename)
        self.reader = csv.DictReader(self.gzfile)

    def next(self):
        return self.reader.next()

    def close(self):
        self.gzfile.close()

    def __iter__(self):
        return self.reader.__iter__()


class npAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ThreadSafeCounter(object):
    def __init__(self):
        self.val = multiprocessing.Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value


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


###########################
### OS stuff_
###########################

def safely_close(fh):
    fh.flush()
    os.fsync(fh.fileno())  # this and f.flush were needed since after file close, file wasn't immediately available.
    fh.close()


def ensure_dir(f):
    '''

    :param f: file or directory name
    :return: no return val, creates dir if it doesnt exist
    '''
    if f is None:
        logging.warning('cannot create directory for None!')
        return
    logging.debug('f:' + f)
    # d = os.path.dirname(f)
    if not os.path.exists(f):
        #        print('d:'+str(d))

        os.makedirs(f)

def ensure_file(fname):
    if not os.path.exists(fname):
        fhandle = open(fname, 'a')
        try:
            os.utime(fname, None)
        finally:
            fhandle.close()


def immediate_subdirs(dir):
    '''
    returns paths of immediate subdirectories of dir (not recursive)
    '''
    return filter(os.path.isdir, [os.path.join(dir, f) for f in os.listdir(dir)])


def files_in_directory(dir):
    '''
    returns paths of files in directory (not recursive)
    '''
    try:
        list = filter(os.path.isfile, [os.path.join(dir, f) for f in os.listdir(dir)])
        return list
    except OSError:
        logging.error('no such directory ' + dir)


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def depth_of_subdir_of_calling_function():
    '''
    this finds the depth of subdirectory in which the caller resides
    :return:
    '''
    path = os.getcwd()
    #   print('path:'+str(path))
    p2 = path.split('trendi_guru_modules')
    #  print('path split on trendigurumodules:'+str(p2))
    if len(p2) < 2:
        print('not in trendi_guru_modules')
    secondhalf = p2[1]
    #   print('secondhalf:'+str(secondhalf))
    cur = secondhalf.split('/')
    #   print('cur:'+str(cur))
    if len(cur) > 1:
        in_subdir_of_trendi_guru_modules = True
    return len(cur) - 1


def get_images_list(dir_url):
    paths_list = files_in_directory(dir_url)
    images_list = []
    # i=0
    for url in paths_list:
        images_list.append(get_cv2_img_array(url))
        # cv2.imshow('1', images_list[i])
        # cv2.waitKey(0)
        # i += 1
    return images_list


def show_parse(filename=None, img_array=None):
    if filename is not None:
        img_array = cv2.imread(filename)
    if img_array is not None:
        # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_array)
        maxVal = np.amax(img_array)
        scaled = np.multiply(img_array, int(255 / maxVal))
        dest = cv2.applyColorMap(scaled, cv2.COLORMAP_RAINBOW)
        return dest
        cv2.imshow("dest", dest)
        cv2.waitKey(0)
    logging.warning('got None as image array from:'+filename)

def shorten_url_googl(long_url):
    url = "https://www.googleapis.com/urlshortener/v1/url"
    querystring = {"key": "AIzaSyCYaOjTMgUKoopLBe3109V3fXIZtOJ8uec"}
    payload = json.dumps({"longUrl": long_url})
    headers = {'content-type': 'application/json'}
    response = requests.request("POST", url, data=payload, headers=headers, params=querystring)
    return response.json().get("id") or long_url


def shorten_url_bitly(long_url):
    url = "https://api-ssl.bitly.com/v3/shorten"
    querystring = {"access_token": "1b131dcc7af91f1fa7f481ab7c20da0f658acff9",
                   "longUrl": long_url,
                   "format": "txt"}
    response = requests.request("GET", url, params=querystring)
    return response.text.rstrip()


def get_files_from_dir_and_subdirs(path=None):
    '''
    this returns the full paths of every file in and under the given dir
    :param path:
    :return: list of full paths of files (not dirs)
    '''

    if path is None:
        path = os.getcwd()
 #   print('basepath:' + path)
#    raw_input('enter to continue')
    file_list = []
    done_paths = []
    for paths, dirs, files in os.walk(path):
        if paths not in done_paths:
            count = paths.count('/')
            if files:
                for ele1 in files:
#                    raw_input('enter to continue')
                    #                  print('---------' * (count), ele1)
                    full_name = os.path.join(path, ele1)
#                    print('filename:' + str(full_name))
                    file_list.append(full_name)
                print('{0} files found in dir {1}'.format(len(files),path))
            if dirs:
                for ele2 in dirs:
#                    print('dir:'+str( ele2))
                    abs_path = os.path.join(paths, ele2)
                    # recursively calling the direct function on each directory
                    more_files = get_files_from_dir_and_subdirs(path=abs_path)
                    # adding the paths to the list that got traversed
                    done_paths.append(abs_path)
                    for file_n in more_files:
                        file_list.append(file_n)
    return(file_list)

def remove_duplicate_files(dir):
    '''
    remove dupe files from dir  - warning this deletes files
    :param dir:
    :return: number of dupes removed
    '''
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    print('n files:'+str(len(files)))
    hashes = []
    dupe_count = 0
    for a_file in files:
        fullname = os.path.join(dir,a_file)
#        img_arr = cv2.imread(fullname)
        with open(fullname,'r') as f:
            logging.debug('current file:'+fullname)
            contents = f.read()
            if contents is not None:
                m = hashlib.md5()
                m.update(contents)
                current_hash = m.hexdigest()
                logging.debug('image hash:' + current_hash + ' for ' + a_file)
                dupe_flag = False
                for a_previous_hash in hashes:
                    if  current_hash == a_previous_hash:
                        fullpath = os.path.join(dir,a_file)
                        print('going to remove '+str(fullpath))
                        os.remove(fullpath)
                        dupe_flag = True
                        dupe_count = dupe_count + 1
                        break
                if not dupe_flag:
                    hashes.append(current_hash)
                    print(fullname+' not a dupe')
    print('found {} dupes'.format(dupe_count))


# testing git pull on pp2
def git_pull(**kwargs):
    import subprocess
    path = os.path.abspath(__file__)
    module_directory = os.path.dirname(path)
    print("Git_pull pulling to: " + module_directory)
    try:
        result = subprocess.check_output('git -C {dir} pull'.format(dir=module_directory), shell=True)
    except subprocess.CalledProcessError, e:
        #        logging.warning("git_pull failed with exception: {0}\ngit output:{1}".format(e))   #needs the second field
        logging.warning("git_pull failed with exception: {0}".format(e))
    return

def map_function_on_dir(func,dirname,**arglist):
    '''
    takes a function that has a filename as first arg and maps it onto files in dirname
    :param func: function to map
    :param dirname: dir of files to do function on
    :param arglist: args to func
    :return:
    '''
    logging.debug('applying function {} to files in directory {} with arguments {}'.format(func,dirname,str(arglist)))
    only_files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    for a_file in only_files:
        fullpath = os.path.join(dirname,a_file)
        func(fullpath,arglist)

def map_function_on_dir_of_dirs(func,dir_of_dirs,**arglist):
    '''
    takes a function that has a filename as first arg and maps it onto files in directory of directories
    :param func: function to map
    :param dir_of_dirs: dir of dirs to do function on
    :param arglist: args to func
    :return:
    '''
    logging.debug('applying function {} to files in directories under directory {} with arguments {}'.format(func,dir_of_dirs,str(arglist)))
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    for a_dir in only_dirs:
        fullpath = os.path.join(dir_of_dirs,a_dir)
        map_function_on_dir(func,fullpath,**arglist)


def parent_dir(path):
    '''
    returns parent of file or dir pointed to by path
    :param path:
    :return: parent
    '''
    return os.path.abspath(os.path.join(path, os.pardir))



############################
### math stuff
############################

def precision(true_pos=0, false_pos=0):
    if true_pos + false_pos:
        precision = float(true_pos) / (true_pos + false_pos)
    else:
        precision = 0
    return precision


def recall(true_pos=0, false_neg=0):
    if true_pos + false_neg:
        recall = float(true_pos) / (true_pos + false_neg)
    else:
        recall = 0
    return recall


def intersectionOverUnion(r1, r2):
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


def error_of_fraction(numerator, numerator_stdev, denominator, denominator_stdev):
    """
    this gives the error on fraction numerator/denominator assuming no covariance
    :param numerator:
    :param numerator_stdev:
    :param denominator:
    :param denominator_stdev:
    :return:
    """
    n = float(numerator)
    d = float(denominator)
    n_e = float(numerator_stdev)
    d_e = float(denominator_stdev)
    if n == 0 or d == 0:
        print('caught div by zero in error_of_fraction, n=' + str(n) + ' d=' + str(d))
        return (-1.0)
    fraction_error = abs(n / d) * math.sqrt((n_e / n) ** 2 + (d_e / d) ** 2)
    return fraction_error


def lines_in_file(filename):
    line_count = 0
    try:
        with open(filename, 'r') as fp:
            for line in fp:
                line_count = line_count + 1
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        logging.error('oops. an environment error. take cover!!! ' + str(sys.exc_info()[0]))
        return line_count
    return line_count


def isnumber(str):
    num_format = re.compile("^[1-9][0-9]*\.?[0-9]*")
    isnumber = re.match(num_format, str)
    if isnumber:
        return True
    else:
        return False


def kick_fp_out():
    fp = 'people.0.items.0.similar_results.0.fingerprint'
    idx = 0
    for doc in db.images.find({fp: {'$exists': 1}}):
        print("started")
        idx += 1
        for person in doc['people']:
            for item in person['items']:
                for result in item['similar_results']:
                    if 'fingerprint' in result.keys():
                        result.pop('fingerprint')
        db.images.replace_one({'_id': doc['_id']}, doc)
        print("did {0} docs".format(idx))
    print("{0} docs modified".format(idx))


def data_url_to_cv2_img(url):
    nparr = np.fromstring(url.split(',')[1].decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


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

def flatten_list(l):
    flatlist = []
    for sublist in l:
        if isinstance(sublist,list):
            flattened  = flatten_list(sublist)
 #           print('flattened sublist:'+str(flattened))
            flatlist = flatlist + flattened
        else:
            flatlist.append(sublist)
    return flatlist

def give_me_a_list_of_synonyms(category,synonym_lists=constants.synonymous_cats):
    '''
    this takes a synonymlist of list and returns the relevant list if any
    synonym list is
#    synonymous_cats =[  ['womens_swimwear_nonbikini','women\'s swimwear'],
 #                   ['leggings','stockings'] ... ]
    :param category:
    :return:
    '''

    for l in synonym_lists:
        if category in l:
            return l
    return [category]  #no list available so return just the category itself

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


if __name__ == '__main__':
    print('starting')
    # show_all_bbs_in_db()
    # fix_all_bbs_in_db()
    # step_thru_db(use_visual_output=True)
    # http://glamradar.com/wp-content/uploads/2013/01/Chiffon-Maxi-Skirts-celebs-500x500.jpg
    show_parse(
        '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/paperdoll/glamradar.com_wp-content_uploads_2013_01_Chiffon-Maxi-Skirts-celebs-500x500.png')
