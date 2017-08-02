from __future__ import print_function

import multiprocessing

__author__ = 'liorsabag and jeremyrutman'
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


if __name__ == '__main__':
    print('starting')
    # show_all_bbs_in_db()
    # fix_all_bbs_in_db()
    # step_thru_db(use_visual_output=True)
    # http://glamradar.com/wp-content/uploads/2013/01/Chiffon-Maxi-Skirts-celebs-500x500.jpg
    show_parse(
        '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/paperdoll/glamradar.com_wp-content_uploads_2013_01_Chiffon-Maxi-Skirts-celebs-500x500.png')
