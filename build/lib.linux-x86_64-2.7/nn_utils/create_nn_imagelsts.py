__author__ = 'jeremy'
import numpy as np
import os
import cv2
import random
import logging
import sys
from shutil import copyfile
import json
import shutil
import pymongo
import itertools

logging.basicConfig(level=logging.INFO)
from PIL import Image

from trendi import constants
from trendi.utils import imutils
from trendi import Utils
from trendi.yonatan import yonatan_constants
from trendi.features import config

def verify_files_in_list_exist(filelist):
    if not os.path.exists(filelist):
        print('did not find {}'.format(filelist))
        return
    with open(filelist,'r') as fp:
        lines = fp.readlines()
    print('{} lines in {}'.format(len(lines),filelist))
    n=0
    badlist = []
    for line in lines:
        if n%100==0:
            print('checked {}/{}'.format(n,len(lines)))
        file = line.strip('\n')
        if not os.path.exists(file):
            print('{} does not exist!!!'.format(file))
            badlist.append(file)
        n=n+1
    print('list of bad files:{}'.format(badlist))

def write_cats_from_db_to_textfile(image_dir='/data/jeremy/image_dbs/tamara_berg/images',catsfile = 'tb_cats_from_webtool.txt'):
    '''
    for tamara berg cats
    todo - put in consistency check , ie make sure at least 2 votes for 'yes' and  0 votes for 'no' as in binary_pos_and_neg_from_multilabel_db
    :param image_dir:
    :param catsfile:
    :return:
    '''
    db = constants.db
    cursor = db.training_images.find()
    n_done = cursor.count()
    print(str(n_done)+' docs in db')
    lines_written = 0
    n_consistent = 0
    n_inconsistent = 0
    min_votes_for_positive=2
    max_votes_for_negative=0
    with open(catsfile,'w') as fp:
        for i in range(n_done):
            document = cursor.next()
            url = document['url']
            filename = os.path.basename(url)
            full_path = os.path.join(image_dir,filename)
            items_list = document['items'] #
            hotlist = np.zeros(len(constants.web_tool_categories_v2))
            if not 'already_seen_image_level' in document:
                print('no votes for this doc')
                continue
            if document['already_seen_image_level'] < 2:
                print('not enough votes for this doc')
                continue
            for item in items_list:
                cat = item['category']
                if cat in constants.web_tool_categories_v2:
                    index = constants.web_tool_categories_v2.index(cat)
                elif cat in constants.tamara_berg_to_web_tool_dict:
                    print('WARNING translating from TB')
                    raw_input('WARNING')
                    cat = constants.tamara_berg_to_web_tool_dict[cat]
                    index = constants.web_tool_categories_v2.index(cat)
                else:
                    print('could not figure out this category : '+str(cat))
                    if cat == 'blazer':
                        index = constants.web_tool_categories_v2.index('jacket')
                        print('replacing blazer with jacket ( cat {}) '.format(index))
                    continue
                hotlist[index] = hotlist[index]+1
#                print('item:'+str(cat))

            consistent=all([(elem>=min_votes_for_positive or elem<=max_votes_for_negative) for elem in hotlist])
            n_consistent = n_consistent + consistent
            n_inconsistent = n_inconsistent + int(not(consistent))
            print('consistent:'+str(consistent)+' n_con:'+str(n_consistent)+' incon:'+str(n_inconsistent))
            print('hotlist:'+str(hotlist))
            if(consistent):
                line = str(full_path) +' '+ ' '.join(str(int(n)) for n in hotlist)
                lines_written +=1
                fp.write(line+'\n')
    print(str(lines_written)+' lines written to '+catsfile)

def consistency_check_multilabel_db(in_docker=True):
    '''
    read multilabel db, tally up total tags
    check images that have been gone over by 2 or more ppl
    do something about disagreements
    '''
    n_consistent = 0
    n_inconsistent = 0
    min_votes_for_positive=2
    max_votes_for_negative=0
    print('attempting db connection')
    if in_docker:
        db = pymongo.MongoClient('localhost',port=27017).mydb
    else:
        db = constants.db
    cursor = db.training_images.find()
    n_total = cursor.count()
    print(str(n_total)+' docs total')
    all_items_dict = {}
    for document in cursor:
#    for i in range(n_total):
#        document = cursor.next()
#        print(document)
        items_list = document['items']
        if items_list is None:
            print('no items in doc')
            continue
        totlist = {}
        for item in items_list:
            cat = item['category']
#            print('cat:'+str(cat))
            if cat in constants.web_tool_categories_v2:
#                print('cat in webtool cats v2')
                pass
            elif cat in constants.tamara_berg_to_web_tool_dict:
#                print('cat in tamara_ber_to_webtool_dict')
                pass
            else:
                print('unrecognized cat')
            if cat in totlist:
                totlist[cat] += 1
            else:
                totlist[cat] = 1
        if totlist == {}:
            print('totlist is {}')
            continue
        print('totlist:'+str(totlist))
        cat_totals = [totlist[cat] for cat in totlist]
#        print('cat totals:'+str(cat_totals))
#         if cat_totals[0] == 1:
#             consistent = False
#         else:
#             consistent = cat_totals and all(cat_totals[0] == elem for elem in cat_totals)
        consistent=all([(elem>=min_votes_for_positive or elem<=max_votes_for_negative) for elem in cat_totals])
        if consistent:
            for key, value in totlist.iteritems():
                if key in all_items_dict:
                    all_items_dict[key] += 1
                else:
                    all_items_dict[key] = 1

        print('all_items_dict:' + str(all_items_dict))
        n_consistent = n_consistent + consistent
        n_inconsistent = n_inconsistent + int(not(consistent))
        print('consistent:'+str(consistent)+' n_con:'+str(n_consistent)+' incon:'+str(n_inconsistent))
        print
    # print('cat_totals:'+str(cat_totals)+' totlist:'+str(totlist))
    print('all_items_dict:' + str(all_items_dict))
    print('consistent:'+str(consistent)+' n_con:'+str(n_consistent)+' incon:'+str(n_inconsistent))

def tg_positives(folderpath='/data/jeremy/image_dbs/tg/google',path_filter='kept',allcats=constants.flat_hydra_cats,outsuffix='pos_tg.txt'):
    '''
    take the tg positives for all cats and put into labelfiles
    :param folderpath:
    :param path_filter:
    :param allcats:
    :param outsuffix:
    :return:
    '''
    for cat in allcats:
        all_filters = [path_filter,cat]
        class_number = 1
        outfile = cat+'_'+outsuffix
        path_antifilter = None
        if cat == 'suit':
            path_antifilter = ['tracksuit','bodysuit']
        if cat == 'bikini':
            path_antifilter = ['nonbikini']
        dir_to_labelfile(folderpath,class_number,outfile=outfile,filefilter='.jpg',path_filter=all_filters,path_antifilter=path_antifilter,recursive=True)

def binary_pos_and_neg_deepfashion_and_mongo(allcats=constants.flat_hydra_cats,outfile='pos_neg_mongo_df.txt'):
    '''
    #1. tamarab berg - generates pos and neg per class
        assume this is already done e.g. using   binary_pos_and_neg_from_multilabel_db
    #2. deepfashion - use constants.bad_negs_for_pos to generate negs for given pos
    #3. mongo db images - again use constants.bad_negs
    (#4. google open images)
    :param cats:
    :return:
    '''
# bailing on tg images for negatives right  now since they are messy - unwanted folders etc.
    #todo - include the tg stuff in negatives (nice to have - but 'only' 50k more negatives)
#    folderpath_tg='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/img_256x256'
#    dirs_and_cats_tg = os_walk_to_tg_hydra(folderpath=folderpath_deepfashion)
#todo deal with substring problem - e.g. suit is a substring of swimsuit and so swimsuit directory can get classed as suit...

    folderpath_deepfashion='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/img_256x256'
    dirs_and_cats_deepfashion = deepfashion_to_tg_hydra(folderpath=folderpath_deepfashion)
    folderpath_mongo='/data/jeremy/image_dbs/mongo'
    dirs_and_cats_mongo = dir_of_dirs_to_tg_hydra(folderpath=folderpath_mongo,cats = allcats)

    for cat in allcats:
        print('generating pos and neg for cat:'+str(cat))
        positives_df,negatives_df = binary_pos_and_neg_using_neglogic_onecat(cat,dirs_and_cats_deepfashion,allcats=allcats,folderpath=folderpath_deepfashion,outfile=cat+'_pos_neg_df.txt')
        positives_mongo,negatives_mongo = binary_pos_and_neg_using_neglogic_onecat(cat,dirs_and_cats_mongo,allcats=allcats,folderpath=folderpath_mongo,outfile=cat+'_pos_neg_mongo.txt')
        allpositives = positives_df+positives_mongo
        allnegatives = negatives_df+negatives_mongo
        filename=cat+'_'+outfile
        print('pos df {} pos mongo {} all {} neg fd {} neg mongo {} tot {} writng to {}'.format(len(positives_df),len(positives_mongo),len(allpositives),len(negatives_df),len(negatives_mongo),len(allnegatives),filename))
        with open(filename,'a') as fp:
            for positive in allpositives:
                fp.write(str(positive)+'\t1\n')
            for negative in allnegatives:
                fp.write(str(negative)+'\t0\n')
        raw_input('ret to cont')

def binary_pos_and_neg_using_neglogic_onecat(cat,dirs_and_cats,allcats=constants.flat_hydra_cats,folderpath='/data/jeremy/image_dbs/mongo',outfile=None):
    '''
    given a category and list of directories with a known cat per dir , use constants.bad_negs_for_pos to determine what cant be used as negs
    for the cat - generate negs using everything else not in bad_negs, generate pos using the cat , write to file and return lists
    :param cat:
    :param dirs_and_cats:
    :param allcats:
    :param folderpath:
    :param outfile:
    :return:
    '''
    if cat is 'None' or cat is None:
        logging.warning('got none as a cat in binary_pos_and_neg_df_onecat')
        return

    print('got {} dirs/cats, first is {}'.format(len(dirs_and_cats),dirs_and_cats[0]))
    print('looking for cats:'+str(cat))

    #do positives
    positives = []
    cat_synonyms = Utils.give_me_a_list_of_synonyms(cat,constants.synonymous_cats)

    print('category {} synonyms {}'.format(cat,cat_synonyms))
    for d_and_c in dirs_and_cats:
        cat_for_dir = d_and_c[1]
        if cat_for_dir is None:
            logging.info('none cat for dc {}'.format(d_and_c))
            continue
        print('checking dir/cat {}'.format(d_and_c))
        for catsyn in cat_synonyms:
            if catsyn in cat_for_dir: #this directory is a category of interest for positives
                full_path = os.path.join(folderpath,d_and_c[0])
                files = os.listdir(full_path)
                for file in files:
                    file_path = os.path.join(full_path,file)
                    logging.debug('file {} cat {}'.format(file_path,catsyn))  #add no-cr
                    positives.append(file_path)
            break  #no need to go thru rest of the synonyms.
    print('found {} positives for cat {} (using sysnonyms {})'.format(len(positives),cat,cat_synonyms))

    #do negatives
    negatives = []
    if cat in  constants.bad_negs_for_pos:
        dont_use_these_neg_cats = constants.bad_negs_for_pos[cat]
        dont_use_these_neg_cats=Utils.flatten_list(dont_use_these_neg_cats)
    else:
        logging.warning('could not find cat {} in constants.bad_negs_for_pos'.format(cat))
        return positives,negatives

    print('bad negs for cat {}:\n{}'.format(cat,dont_use_these_neg_cats))
    print
    for potential_negative in allcats:
        negative_shouldnt_be_used_flag = False
        if potential_negative == cat:
            continue #dont kill the cat under consideration
        pot_neg_synonyms = Utils.give_me_a_list_of_synonyms(potential_negative,constants.synonymous_cats)
        print('potential neg synonyms:'+str(pot_neg_synonyms))
        for potential_negative_synonym in pot_neg_synonyms:
            if potential_negative_synonym in dont_use_these_neg_cats:
                print('potential neg {} negged '.format(potential_negative_synonym))
                negative_shouldnt_be_used_flag = True
                break
        if negative_shouldnt_be_used_flag:
            logging.info('negative {} shold not be used'.format(potential_negative_synonym))
            continue
        print('not negged and therefore useful as negative for {}:{}'.format(cat,potential_negative))
        for d_and_c in dirs_and_cats:
            cat_for_dir = d_and_c[1]
            if cat_for_dir is None:
                print('none cat for dc {}'.format(d_and_c))
                continue
            if cat_for_dir in pot_neg_synonyms: #this directory is a category of interest
                full_path = os.path.join(folderpath,d_and_c[0])
                files = os.listdir(full_path)
#                print('using dir {} as neg for cat {}, {} files '.format(d_and_c[0],cat,len(files)))
                for file in files:
                    file_path = os.path.join(full_path,file)
                    negatives.append(file_path)
#                    fp.write(file_path+'\t'+str(cat_index)+'\n')
#                    logging.debug('wrote "{} {}" for file {} cat {}'.format(file_path,cat_index,file,cat_index))  #add no-cr
            else:
#                logging.info('catfordir {} not in pot_neg_syn {}'.format(cat_for_dir,pot_neg_synonyms))
                pass
        print('done with negative {}, current size {}'.format(potential_negative,len(negatives)))
        raw_input('ret to cont')
    print('done with all negatives, n_pos {} n_neg {}'.format(len(positives),len(negatives)))
    if outfile is not None:
        with open(outfile,'a') as fp:
            for positive in positives:
                fp.write(str(positive)+'\t1\n')
            for negative in negatives:
                fp.write(str(negative)+'\t0\n')
    return positives, negatives

def dir_of_dirs_to_tg_hydra(folderpath='/data/jeremy/image_dbs/mongo',cats=constants.flat_hydra_cats,filefilter=None):
    '''
    the mongo dbs are downloaded as a folder per db, with subfolders for the categories
    :param folderpath:
    :param cats:
    :return:
    '''
    cats_and_dirs = []

    subdirs = [os.path.join(folderpath, name) for name in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, name)) ]
    for dir in subdirs:
        print('dir:'+dir)
        subsubdirs = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir,name))]
        for subsubdir in subsubdirs:
            print('subsubdir:'+subsubdir)
            cat_for_dir = None
            for cat in cats:
                cat_synonyms = Utils.give_me_a_list_of_synonyms(cat,constants.synonymous_cats)
                print('category {} synonyms {}'.format(cat,cat_synonyms))
                for catsyn in cat_synonyms:
                    if catsyn in subsubdir:
                        cat_for_dir = cat
                        #dont break here! continue all the way and see if there are other matches, take the longest
                        #to avoid the 'substring problem' namely suit matches jumpsuit etc
                        break
                if cat_for_dir is None:
                    print('could not get cat for dir '+str(subsubdir))
                else:
                    full_dirpath = os.path.join(dir,subsubdir)
                    cats_and_dirs.append([full_dirpath,cat_for_dir])
                    print('cat for {} is {}'.format(full_dirpath,cat_for_dir))
                    break
    print('{} cats and dirs '+str(len(cats_and_dirs)))
#    print cats_and_dirs
    return cats_and_dirs

def os_walk_to_tg_hydra(folderpath='/data/jeremy/image_dbs/mongo',cats=constants.flat_hydra_cats,recursive=True,filefilter=None):
    '''
    the mongo dbs are downloaded as a folder per db, with subfolders for the categories
    :param folderpath:
    :param cats:
    :return:
    '''
    cats_and_dirs = []

    if recursive:
        for root,dirs,files in os.walk(folderpath):
            #path = root.split(os.sep)
            print('root {}'.format(root))
            newfiles = [os.path.join(root,f) for f in files]
            if filefilter:
                newfiles = [f for f in newfiles if filefilter in f]
    else:
        newfiles = [os.path.join(folderpath,f) for f in os.listdir(folderpath)]
        if filefilter:
            newfiles = [f for f in newfiles if filefilter in f]

    unique_dirs = []
    for f in newfiles:
        if not os.path.dirname(f) in unique_dirs:
            unique_dirs.append(os.path.dirname(f))
            print('unique dir: '+str(os.path.dirname(f)))
    cats_and_dirs = []
    for dir in unique_dirs:
        print('dir:'+dir)
        cat_for_dir = None
        for cat in cats:
            cat_synonyms = Utils.give_me_a_list_of_synonyms(cat,constants.synonymous_cats)
            print('category {} synonyms {}'.format(cat,cat_synonyms))
            for catsyn in cat_synonyms:
                if catsyn in dir: #this directory is a category of interest for positives
                    cat_for_dir = cat
                    break  #no need to go thru rest of the synonyms.
            if cat_for_dir is None:
                print('could not get cat for dir '+str(dir))
            else:
                cats_and_dirs.append([dir,cat_for_dir])
                print('cat for {} is {}'.format(dir,cat_for_dir))
                break
    print('{} cats and dirs '+str(len(cats_and_dirs)))
#    print cats_and_dirs
    return cats_and_dirs

def binary_pos_and_neg_from_multilabel_db(image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',catsfile_dir = './',in_docker=True):
    '''
    read multilabel db.
    if n_votes[cat] = 0 put that image in negatives for cat.
    if n_votes[cat] >= 2 put that image in positives for cat
    '''

    if in_docker:
        db = pymongo.MongoClient('localhost',port=27017).mydb
    else:
        db = constants.db

    cursor = db.training_images.find()
    n_done = cursor.count()
    print(str(n_done)+' docs done')
    for i in range(n_done):
        document = cursor.next()
        if not 'already_seen_image_level' in document:
            print('no votes for this doc')
            continue
        if document['already_seen_image_level']<2:
            print('not enough votes for this doc')
            continue
        url = document['url']
        filename = os.path.basename(url)
        full_path = os.path.join(image_dir,filename)
        if not os.path.exists(full_path):
            print('file '+full_path+' does not exist, skipping')
            continue
        items_list = document['items'] #
        if items_list is None:
            print('no items in doc')
            continue
        print('items:'+str(items_list))
        votelist = [0]*len(constants.web_tool_categories_v2)
        for item in items_list:
            cat = item['category']
            if cat in constants.web_tool_categories_v2:
                index = constants.web_tool_categories_v2.index(cat)
            elif cat in constants.tamara_berg_to_web_tool_dict:
                print('old cat being translated')
                cat = constants.tamara_berg_to_web_tool_dict[cat]
                index = constants.web_tool_categories.index(cat)
            else:
                print('unrecognized cat')
                continue
            votelist[index] += 1
            print('item:'+str(cat) +' votes:'+str(votelist[index]))
        print('votes:'+str(votelist))
        for i in range(len(votelist)):
            catsfile = os.path.join(catsfile_dir,constants.web_tool_categories_v2[i]+'_filipino_labels.txt')
            print('catsfile:'+catsfile)
            with open(catsfile,'a') as fp:
                if votelist[i]==0:
                    line = str(full_path) + ' 0 \n'
                    print line
                    fp.write(line)
                if votelist[i] >= 2:
                    line = str(full_path) + ' 1 \n'
                    print line
                    fp.write(line)
                fp.close()

def one_class_positives_from_multilabel_db(image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',
                                           catsfile_dir = '/data/jeremy/image_dbs/labels',catsfile=None,
                                           desired_cat='suit',desired_index=6,in_docker=False,check_file_existence=False):
    '''
    read multilabel db.
    if n_votes[cat] >= 2, put that image in positives for cat
    '''
    if in_docker:
        db = pymongo.MongoClient('localhost',port=27017).mydb
    else:
        db = constants.db
    print('attempting db connection')
    cursor = db.training_images.find()
    n_done = cursor.count()
    print(str(n_done)+' docs done')
    Utils.ensure_dir(catsfile_dir)
    if catsfile is None:
        catsfile = os.path.join(catsfile_dir,desired_cat+'_positives.txt')
    else:
        catsfile = os.path.join(catsfile_dir,catsfile)

    print('catsfile:'+catsfile)
    n_items = 0

    for i in range(n_done):
        document = cursor.next()
        if not 'already_seen_image_level' in document:
            print('no votes for this doc')
            continue
        if document['already_seen_image_level']<2:
            print('not enough votes for this doc')
            continue
        url = document['url']
        filename = os.path.basename(url)
        full_path = os.path.join(image_dir,filename)
        if check_file_existence and not os.path.exists(full_path):
            print('file '+full_path+' does not exist, skipping')
            continue
        items_list = document['items'] #
        if items_list is None:
            print('no items in doc')
            continue
#        print('items:'+str(items_list))
        votes_for_item = 0
        for item in items_list:
            cat = item['category']
            if cat==desired_cat:
                votes_for_item+=1
        if votes_for_item>=2:
            print('votes:'+str(votes_for_item))
            n_items += 1
            with open(catsfile,'a') as fp:
                line = str(full_path) + '\t'+str(desired_index)+'\n'
                print line
                fp.write(line)
                fp.close()
    print('number of matches found:'+str(n_items))
    return n_items

def all_positives_from_multilabel_db(image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',
                                           catsfile_dir = '/data/jeremy/image_dbs/labels',
                                           desired_index=1,in_docker=True):
    '''
    read multilabel db.
    if n_votes[cat] >= 2, put that image in positives file - so all images with clothes should get into labelfile once
    useful for relevant/irrelevant, prob nothing else
    consider putting naked ppl into the irrelevants...
    '''

    print('attempting db connection')
    if in_docker:
        db = pymongo.MongoClient('localhost',port=27017).mydb
    else:
        db = constants.db
        #nb this can apparently be done by checking
        #/proc/1/cgroup
        #which has lines with 'docker' instead of starting with just /, as in
        #1:name=systemd:/docker/85cee1c45352bc4814940e486dbcb169c17042d2b7460e4945ea6909b51a6a1b
        #see http://stackoverflow.com/questions/20010199/determining-if-a-process-runs-inside-lxc-docker
    cursor = db.training_images.find()
    n_done = cursor.count()
    print(str(n_done)+' docs in db')
    n_positives=0
    positives_list=[]
    for i in range(n_done):
        catsfile = os.path.join(catsfile_dir,'all_clothes_tb.txt')
        with open(catsfile,'a') as fp:
            document = cursor.next()
            if not 'already_seen_image_level' in document:
                print('no votes for this doc')
                continue
            if document['already_seen_image_level']<2:
                print('not enough votes for this doc')
                continue
            url = document['url']
            filename = os.path.basename(url)
            full_path = os.path.join(image_dir,filename)
            if not os.path.exists(full_path):
                print('file '+full_path+' does not exist, skipping')
                continue
            items_list = document['items'] #
            if items_list is None:
                print('no items in doc')
                continue
            #print('items:'+str(items_list))
            votelist = [0]*len(constants.web_tool_categories_v2)
            for item in items_list:
                cat = item['category']
                if cat in constants.web_tool_categories_v2:
                    index = constants.web_tool_categories_v2.index(cat)
                elif cat in constants.tamara_berg_to_web_tool_dict:
                    print('old cat being translated')
                    cat = constants.tamara_berg_to_web_tool_dict[cat]
                    index = constants.web_tool_categories.index(cat)
                else:
                    print('unrecognized cat')
                    continue
                votelist[index] += 1
                if votelist[index] >=2:
                    line = str(full_path) + '\t'+str(desired_index)+'\n'
                    fp.write(line)
                    print('item:'+str(cat) +' votes:'+str(votelist[index])+' ' +line)
                    n_positives+=1
                    positives_list.append(full_path)
                    break
    fp.close()
    print('tot positives:'+str(n_positives))
    return positives_list

def analyze_negs_filipino_db(labels=constants.multilabel_categories_v2,in_docker=True):
    '''
    TODO - finish this and verify that constants.bad_negs_for_pos is consistent with this info
    namely this function should return what items occur togethe and with what frequency
    anything that shows up togethe should go into constants.bad_negs_for_pos
    :param labels:
    :param in_docker:
    :return:
    '''
    if in_docker:
        db = pymongo.MongoClient('localhost',port=27017).mydb
    else:
        db = constants.db
    cursor = db.training_images.find()
    n_done = cursor.count()
    print(str(n_done)+' docs done')
    fellow_positives = [  [0 for i in len(constants.web_tool_categories_v2)]  for j in len(constants.web_tool_categories_v2)]
    fellow_negatives = [  [0 for i in len(constants.web_tool_categories_v2)]  for j in len(constants.web_tool_categories_v2)]
    for i in range(n_done):
        document = cursor.next()
        if not 'already_seen_image_level' in document:
            print('no votes for this doc')
            continue
        if document['already_seen_image_level']<2:
            print('not enough votes for this doc')
            continue
        items_list = document['items'] #
        if items_list is None:
            print('no items in doc')
            continue
        print('items:'+str(items_list))
        votelist = [0]*len(constants.web_tool_categories_v2)
        for item in items_list:
            cat = item['category']
            if cat in constants.web_tool_categories_v2:
                index = constants.web_tool_categories_v2.index(cat)
            elif cat in constants.tamara_berg_to_web_tool_dict:
                print('old cat being translated')
                cat = constants.tamara_berg_to_web_tool_dict[cat]
                index = constants.web_tool_categories.index(cat)
            else:
                print('unrecognized cat')
                continue
            votelist[index] += 1
            print('item:'+str(cat) +' votes:'+str(votelist[index]))
        print('votes:'+str(votelist))
        for i in range(len(votelist)):
            cat = constants.web_tool_categories_v2[i]
            for j in range(len(votelist)):
                if votelist[i]==0:
                    fellow_negatives[i]+=1

                if votelist[i] >= 2:
                    fellow_positives[i]+=1

#votes [2 1 0 0 2]
#cat 0 fellow_pos : [[100 10 . . .][
#cat 0 fellow_neg : [100 1

def positives_from_tbdb_for_hydra_cats():
    for type in constants.hydra_cats:
        index = 0
        for cat in type:
            if cat is None:
                index += 1
                continue
            print('doing cat {} index {}'.format(cat,index))
            raw_input('ret to cont')
            one_class_positives_from_multilabel_db(desired_cat=cat,desired_index=index)
            index += 1

def create_class_a_vs_class_b_file_from_multilabel_db(index_a,index_b,image_dir='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',outfile=None,labels=constants.web_tool_categories_v2,skip_missing_files=False):
    '''
    read multilabel db.
    if n_votes[cat] = 0 put that image in negatives for cat.
    if n_votes[cat] = n_voters put that image in positives for cat
    dress 4 skirt 12...
    '''
    if outfile is None:
        outfile = 'class'+str(index_a)+'_vs_class'+str(index_b)+'.txt'
    db = constants.db
    cursor = db.training_images.find()
    n_done = cursor.count()
    n_instances=[0,0]
    output_cat_for_a = 0
    output_cat_for_b = 1
    outlines=[]
    print(str(n_done)+' docs to check')
    for i in range(n_done):
        document = cursor.next()
        if not 'already_seen_image_level' in document:
            print('no votes for this doc')
            continue
        if document['already_seen_image_level']<2:
            print('not enough votes for this doc')
            continue
        url = document['url']
        filename = os.path.basename(url)
        full_path = os.path.join(image_dir,filename)
        if not os.path.exists(full_path) and skip_missing_files:
            print('file '+full_path+' does not exist, skipping')
            continue
        items_list = document['items'] #
        if items_list is None:
            print('no items in doc')
            continue
#        print('items:'+str(items_list))
        votelist = [0]*len(constants.web_tool_categories_v2)
        for item in items_list:
            cat = item['category']
            if cat in constants.web_tool_categories_v2:
                index = constants.web_tool_categories_v2.index(cat)
            elif cat in constants.tamara_berg_to_web_tool_dict:
                print('old cat being translated')
                cat = constants.tamara_berg_to_web_tool_dict[cat]
                index = constants.web_tool_categories.index(cat)
            else:
                print('unrecognized cat')
                continue
            votelist[index] += 1
#            print('item:'+str(cat) +' votes:'+str(votelist[index]))
        print('votes:'+str(votelist))
        if votelist[index_a]>=2 and votelist[index_b]==0:
            line = str(full_path) + ' '+str(output_cat_for_a)+'\n'
            n_instances[0]+=1
            print('catA file {} n {}'.format(full_path,n_instances))
            outlines.append(line)
        elif votelist[index_a]==0 and votelist[index_b]>=2:
            line = str(full_path) + ' '+str(output_cat_for_b)+'\n'
            n_instances[1]+=1
            print('catB file {} n {}'.format(full_path,n_instances))
            outlines.append(line)
        else:
            print('{} votes for cat {} and {} votes for cat {} b, not using'.format(votelist[index_a],index_a,votelist[index_b],index_b))
    print('writing {} lines to {}, breakdown:{}'.format(len(outlines),outfile,n_instances))
    with open(outfile,'w') as fp:
        for l in outlines:
            fp.write(l)
        fp.close()

def create_class_a_vs_class_b_file_from_multilabel_file(index_a,index_b,multilabel_textfile,visual_output=False,outfile=None):
    if outfile is None:
        outfile = 'class'+str(index_a)+'_vs_class'+str(index_b)+'.txt'
    n_instances=[0,0]
    output_cat_for_a = 0
    output_cat_for_b = 1
    outlines = []
    with open(multilabel_textfile,'r') as fp:
        for line in fp:
   #         print line
            path = line.split()[0]
            vals = [int(v) for v in line.split()[1:]]
            v1 = vals[index_a]
            v2 = vals[index_b]
            if v1 and v2:
                print('got image {} with both cats, not using'.format(path))
            elif v1:
                n_instances[0]+=1
                outlines.append(path+' '+str(output_cat_for_a))
                print('indexa {} indexb {} file {} n {}'.format(v1,v2,path,n_instances))
            elif v2:
                n_instances[1]+=1
                outlines.append(path+' '+str(output_cat_for_b))
                print('indexa {} indexb {} file {} n {}'.format(v1,v2,path,n_instances))
            else:
                print('got image {} with no cats, not using'.format(path))
            if(visual_output):
                img_arr = cv2.imread(path)
                imutils.resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True)
        fp.close()
    with open(outfile,'a') as f2:
        for line in outlines:
            f2.write(line+'\n')
        f2.close()

def dir_of_dirs_to_labelfiles(dir_of_dirs,class_number=1):
    dirs = [os.path.join(dir_of_dirs,d) for d in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,d))]
    for d in dirs:
        print('doing directory:'+str(d))
        dir_to_labelfile(d,class_number,outfile=os.path.basename(d)+'_labels.txt',filter='.jpg')

def dir_to_labelfile(dir,class_number,outfile=None,filefilter='.jpg',path_filter=None,path_antifilter=None,recursive=False):
    '''
    take a dir and add the files therein to a text file with lines like:
    /path/to/file class_number
    :param dir:
    :param class_number: assign all files this class #
    :param outfile : write to this file.  Appends, doesn't overwrite
    :pathfilter - list of required terms in path e..g male and swimsuit
    :path_antifilter - list of terms that cant occur in path e.g female
    :return:#
    '''
    print('class {} filter {} antifilter {}'.format(class_number,path_filter,path_antifilter))
    if recursive:
        allfiles = []
        for root,dirs,files in os.walk(dir):
            #path = root.split(os.sep)
#            print('root {}'.format(root))
            newfiles = [os.path.join(root,f) for f in files]
            if filefilter:
                newfiles = [f for f in newfiles if filefilter in f]
            if path_filter:
#                newfiles = filter(lambda f: not any([term in f for term in path_antifilter]), newfiles)
                newfiles = filter(lambda f: all([term in f for term in path_filter]), newfiles)
            if path_antifilter:
                newfiles = filter(lambda f: not any([term in f for term in path_antifilter]), newfiles)
            if len(newfiles)>0:
                print('root {}, {} newfiles , filter {} antifilter {}'.format(root,len(newfiles),path_filter,path_antifilter))
            allfiles += newfiles
 #       raw_input('ret to cont')
    else:
        allfiles = [os.path.join(dir,f) for f in os.listdir(dir)]
        if filefilter:
            allfiles=[f for f in allfiles if filefilter in f]
        if path_filter:
            allfiles = filter(lambda f: all([term in f for term in path_filter]), allfiles)
        if path_antifilter:
            allfiles = filter(lambda f: not any([term in f for term in path_antifilter]), allfiles)
    i = 0
    if outfile == None:
        outfile = os.path.join(dir,'labelfile.txt')
    if len(allfiles) == 0:
        print('didnt find any files so not writing')
        return allfiles
    with open(outfile,'a') as fp:
        for f in allfiles:
            line = f + '\t'+str(class_number)
            logging.debug(line)
            fp.write(line+'\n')
            i+=1
        fp.close()
    print('added {} files to {} with class {}'.format(len(allfiles),outfile,class_number))
#    print('dir {} with {} files'.format(dir,len(os.listdir(dir))))
    print(str(i)+' images written to '+outfile+' with label '+str(class_number))
    print('')
    return allfiles


def copy_negatives(filename = 'tb_cats_from_webtool.txt',outfile =  None):
    '''
    file lines are of the form /path/to/file class_number
    :param filename:
    :return:
    '''
    negs = []
    if outfile == None:
        outfile = filename[:-4]+'_negs.txt'
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            path = line.split()[0]
            cat = int(line.split()[1])
            if cat == 0:
                negs.append(line)
        fp.close()
    print('n_negatives {}'.format(len(negs)))

    if filename:
        with open(outfile,'w') as fp:
            for line in negs:
                fp.write(line)
    return negs

def inspect_single_label_textfile(filename = 'tb_cats_from_webtool.txt',visual_output=False,randomize=False,cut_the_crap=False):
    '''
    file lines are of the form /path/to/file class_number
    analysis of avg image sizes, rgb values and other stats (per class if so desired) can be easily done here
    :param filename:
    :return:
    '''
    n_instances = {}
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            path = line.split()[0]
            cat = int(line.split()[1])
            if cat in n_instances:
                n_instances[cat]+=1
            else:
                n_instances[cat] = 1
        fp.close()

    print('n_instances {}'.format(n_instances))
    if randomize:
        random.shuffle(lines)
    if n_instances == {}:
        return
    n = 0
    cats_used = [k for k,v in n_instances.iteritems()]
    n_cats = np.max(cats_used) + 1 # 0 is generally a cat so add 1 to get max
    n_encountered = [0]*n_cats
    if visual_output:
        for line in lines:
            n = n + 1
            print line
            path = line.split()[0]
            cat = int(line.split()[1])
            n_encountered[cat]+=1
            print(str(n)+' images seen, totals:'+str(n_encountered))
    #            im = Image.open(path)
    #            im.show()
            img_arr = cv2.imread(path)
            imutils.resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True)
            if cut_the_crap:  #move selected to dir_removed, move rest to dir_kept
                print('(d)elete (c)lose anything else keeps')
                indir = os.path.dirname(path)
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

def inspect_dir(dir,curate=True,remove_parens=True,add_jpg=True,display_size=(700,700)):
    '''
    better version at imutils.image_chooser
    look at images in dir , remove at will
    :param dir:
    :param curate: move checked images to '_kept' dir and deleted to '_removed'
    :return:
    '''
    files = [f for f in os.listdir(dir)]
    n=1

    for file in files:
        print('file {} of {} {}'.format(n,len(files),file))
        filepath = os.path.join(dir,file)
        img_arr = cv2.imread(filepath)
        if img_arr is None:
            print('got no img for '+filepath)
            continue
 #       img_arr = imutils.resize_to_max_sidelength(img_arr, max_sidelength=1000,use_visual_output=False)
        img_arr_resized = imutils.resize_keep_aspect(img_arr,output_size=display_size,use_visual_output=False)
        cv2.imshow('img',img_arr_resized)
        k=cv2.waitKey(0)
        if curate:  #move selected to dir_removed, move rest to dir_kept
            print('(d)elete, anything else keeps')
#            parentdir = os.path.abspath(os.path.join(indir, os.pardir))
#            curdir = os.path.split(indir)[1]
#            print('in {} parent {} cur {}'.format(indir,parentdir,curdir))
            if k == ord('d'):
                newdir = 'removed'
                dest_dir = os.path.join(dir,newdir)
                print('REMOVING moving {}\nto {}'.format(filepath,dest_dir))
                Utils.ensure_dir(dest_dir)
                shutil.move(filepath,dest_dir)
            else:
                newdir = 'kept'
                dest_dir = os.path.join(dir,newdir)
                Utils.ensure_dir(dest_dir)
                if add_jpg:
                    file=file+'.jpg'
                if remove_parens:
                    newfile = file.replace('(','').replace(')','')
                    dest_dir = os.path.join(dest_dir,newfile)
                print('KEEPING moving {}\nto {}'.format(filepath,dest_dir))
                shutil.move(filepath,dest_dir)


        n = n + 1

def inspect_multilabel_textfile(filename = 'tb_cats_from_webtool.txt'):
    '''
    for 'multi-hot' labels of the form 0 0 1 0 0 1 0 1
    so file lines are /path/to/file 0 0 1 0 0 1 0 1
    :param filename:
    :return:
    '''
    with open(filename,'r') as fp:
        for count, line in enumerate(fp):
            print line
            path = line.split()[0]
            vals = [int(i) for i in line.split()[1:]]
            non_zero_idx = np.nonzero(vals)
            print non_zero_idx
            for i in range(len(non_zero_idx[0])):
                print yonatan_constants.attribute_type_dict[str(non_zero_idx[0][i])]

                # img_arr = cv2.imread(os.path.join("/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction", path))
                # if img_arr is None:
                #     print('could not grok file '+path)
                #     continue

                # cv2.imshow("image", img_arr)
                # cv2.waitKey(0)

def inspect_pixlevel_textfile(filename = 'images_and_labelsfile.txt',labels=constants.ultimate_21,replace_this=None,with_this=None):
    with open(filename,'r') as fp:
        for line in fp:
            print line
            path1 = line.split()[0]
            path2 = line.split()[1]
            if replace_this and with_this:
                # base=os.path.basename(path2)
                # path2=os.path.join(replace_pardir,base)
                path1=path2.replace(replace_this,with_this)
                path2=path2.replace(replace_this,with_this)
                print('new paths '+str(path1)+' ,'+str(path2))
            imutils.show_mask_with_labels(path2,labels=labels,original_image=path1,visual_output=True)

def inspect_single_label_textfile(filename = 'tb_cats_from_webtool.txt',visual_output=False,randomize=False,cut_the_crap=False):
    '''
    file lines are of the form /path/to/file class_number
    analysis of avg image sizes, rgb values and other stats (per class if so desired) can be easily done here
    :param filename:
    :return:
    '''
    n_instances = {}
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            path = line.split()[0]
            try:
                cat = int(line.split()[1])
                if cat in n_instances:
                    n_instances[cat]+=1
                else:
                    n_instances[cat] = 1
            except:
                print('lines maybe have no class?')
                print(line)
                return
        fp.close()

    print('n_instances {}'.format(n_instances))
    if randomize:
        random.shuffle(lines)
    if n_instances == {}:
        return
    n = 0
    cats_used = [k for k,v in n_instances.iteritems()]
    n_cats = np.max(cats_used) + 1 # 0 is generally a cat so add 1 to get max
    n_encountered = [0]*n_cats
    if visual_output:
        for line in lines:
            n = n + 1
            print line
            path = line.split()[0]
            cat = int(line.split()[1])
            n_encountered[cat]+=1
            print(str(n)+' images seen, totals:'+str(n_encountered))
    #            im = Image.open(path)
    #            im.show()
            img_arr = cv2.imread(path)
            if img_arr is None:
                logging.warning('could not read '+str(path))
                continue
            imutils.resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True)
            if cut_the_crap:  #move selected to dir_removed, move rest to dir_kept
                print('(d)elete (c)lose anything else keeps')
                indir = os.path.dirname(path)
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

def split_to_trainfile_and_testfile(filename='tb_cats_from_webtool.txt', fraction=0.05,shuffle=True):
    '''
    writes (destructively) files with _train.txt and _test.txt based on filename, with sizes determined by fraction
    :param filename: input catsfile
    :param fraction: ratio test:train
    :return:
    '''
    with open(filename,'r') as fp:
        lines = fp.readlines()
        if lines == []:
            logging.warning('nothing in {}'.format(filename))
            return
        print('file {} has lines like {}'.format(filename,lines[0]))
        if shuffle:
            random.shuffle(lines)
        n_lines = len(lines)
        train_lines = lines[0:int(n_lines*(1-fraction))]
        test_lines = lines[int(n_lines*(1-fraction)):]
        train_name = filename[0:-4] + '_train.txt'
        test_name = filename[0:-4] + '_test.txt'
        print('{} files written to {} and {} files written to {}'.format(len(train_lines),train_name,len(test_lines),test_name))
        with open(train_name,'w') as trfp:
            trfp.writelines(train_lines)
            trfp.close()
        with open(test_name,'w') as tefp:
            tefp.writelines(test_lines)
            tefp.close()
    #report how many in each class
        inspect_single_label_textfile(filename = train_name,visual_output=False,randomize=False)
        inspect_single_label_textfile(filename = test_name,visual_output=False,randomize=False)

def balance_cats(filename='tb_cats_from_webtool.txt', ratio_neg_pos=1.0,n_cats=2,outfilename=None,shuffle=True):
    '''
    balance the occurence of categories - take minimum occurences and let all cats occur only that amt
    ie. if there are 10 examples of class 1, 20 examples class 2, 30 examples class 3, take 10 examples of each class and write
    to outfilename
    there is a theorectical question here of whether this is desireable or not (maybe unbalanced is good if wild is unbalanced)
    this works only for 2 cats (todo - make it work for n cats).  also , assumes there are more negs than pos
    :param filename: input file with lines of the form '/path/to/file  class_number'
    :param ratio_neg_pos: number of negs vs. positives to include , n_neg = n_pos*ratio_neg_pos
    :param outfilename file to write to, if not given writes to original path of catsfile.txt but with filename catsfile.balanced.txt
    :param n_cats not implemented , assumes n_cats=2
    :param shuffle not implemented
    :return:
    '''
    print('balancing '+filename+' with ratio '+str(ratio_neg_pos)+', '+str(n_cats)+' categories')
    n_instances = [0]*n_cats
    instances = []  #*n_cats#iniitialize in Nones . there seems to be no oneliner like instances = [] * n_cats
    for i in range(n_cats):
        instances.append([])
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            path = line.split()[0]
            cat = int(line.split()[1])
            try:
                n_instances[cat]+=1
            except:
                print "Unexpected error:", sys.exc_info()[0]
                print('trying to parse line:')
                print(line)
                print('cat = '+str(cat))
                continue
            instances[cat].append(line)
#        print('path {} cat {} n_instances {}'.format(path,cat,n_instances,instances))
        fp.close()
        print('n_instances {}'.format(n_instances))
    n_negs = n_instances[0]
    n_pos = n_instances[1]
    min_instances = min(n_instances)
    desired_negs = (n_pos*ratio_neg_pos)
    negs_to_use = int(min(desired_negs,n_negs))
    #kill the initial Nones
#    for i in range(n_cats):
#        del(instances[i][0])
#  a shuffle cant hurt here
    if outfilename is None:
        outfilename = filename.replace('.txt','')+'_balanced.txt'
    print('writing {} positives and {} negatives to {}'.format(n_pos,negs_to_use,outfilename))
#    if(shuffle):
#        instances
    with open(outfilename,'w') as fp:
        for i in range(n_cats):
            if i==1:
                jrange=min_instances
            else:
                jrange=negs_to_use
            for j in range(jrange):
                fp.write(instances[i][j])
            print('wrote '+str(jrange)+' lines for category '+str(i))
    fp.close()

def textfile_for_pixlevel(imagesdir,labelsdir=None,imagefilter='.jpg',labelsuffix='.png', outfilename = None):
    if labelsdir == None:
        labelsdir = imagesdir
    if outfilename == None:
        outfilename = os.path.join(imagesdir,'images_and_labelsfile.txt')
    imagefiles = [f for f in os.listdir(imagesdir) if imagefilter in f]
    print(str(len(imagefiles))+' imagefiles found in '+imagesdir)
    with open(outfilename,'w') as fp:
        for f in imagefiles:
            labelfile = f[:-4]+labelsuffix
            labelfile = os.path.join(labelsdir,labelfile)
            if not os.path.exists(labelfile):
                logging.debug('could not find labelfile {} corresponding to imagefile {}'.format(labelfile,f))
                continue
            imagefile = os.path.join(imagesdir,f)
            line = imagefile +' '+ labelfile
            print('writing: '+line)
            fp.write(line+'\n')

def textfile_for_pixlevel_kaggle(imagesdir,labelsdir=None,imagefilter='.tif',labelsuffix='_mask.tif', outfilename = None):
    if labelsdir == None:
        labelsdir = imagesdir
        imagefiles = [f for f in os.listdir(imagesdir) if imagefilter in f and not labelsuffix in f]
    else:
        imagefiles = [f for f in os.listdir(imagesdir) if imagefilter in f]

    if outfilename == None:
        outfilename = os.path.join(imagesdir,'images_and_labelsfile.txt')
    print(str(len(imagefiles))+' imagefiles found in '+imagesdir)
    with open(outfilename,'w') as fp:
        for f in imagefiles:
            labelfile = f[:-4]+labelsuffix
            labelfile = os.path.join(labelsdir,labelfile)
            if not os.path.exists(labelfile):
                logging.debug('could not find labelfile {} corresponding to imagefile {}'.format(labelfile,f))
                continue
            imagefile = os.path.join(imagesdir,f)
            line = imagefile +' '+ labelfile
            print('writing: '+line)
            fp.write(line+'\n')

def deepfashion_folder_to_cat(dir_to_cat,dir):
    for tup in dir_to_cat:
        if tup[0]==dir:
   #         print('matched {} to {}'.format(dir,tup[1]))
            return tup[1]
    print('coulndt find match for '+str(dir))
    return None

def deepfashion_to_tg_hydra(folderpath='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/img'):
    '''
    generate list of deepfashion folders and corresponding categories from our list (map is using constants.deep_fashion_to_trendi_map)
    :param folderpath: folder of deep fashion folders
    :return: list of tuples [(dir1,cat1),(dir2,cat2),...]
    '''
    dirs = os.listdir(folderpath)
    all_cats = []
    for dir in dirs:
        cats_found = []
        for k,v in constants.deep_fashion_to_trendi_map.iteritems():
            if k.lower() in dir.lower():
                if 'velveteen' in dir.lower() and k.lower=='tee': #'velveteen tee' will  get skipped
                    logging.info('skipping dir {} cat {}'.format(dir,k))
                    continue
                else:
                    cats_found.append((dir,v))

                logging.debug('dir {} cat {} match'.format(dir,k))
            else:
                logging.debug('dir {} cat {} NO match'.format(dir,k))
        logging.info('cats for dir {} are {}'.format(dir,cats_found))

        #take care of all the cases where multiple cats are found....yeesh
        cats_found.sort(key=len)
        if len(cats_found)==2 :
            logging.info('dir {} 2 matching cats:{}'.format(dir,cats_found))
            if cats_found[0][1]==cats_found[1][1]:
#                print('matching cats')
                cats_found=[(dir,cats_found[0][1])]
                logging.info('final disposition:'+str(cats_found))
            else:
#                print('nonmatching cats')

                if cats_found[0][1]=='tank' and cats_found[1][1]=='dress':
                    cats_found=[(dir,'dress')]
                elif cats_found[0][1]=='sweater' and cats_found[1][1]=='tank':
                    cats_found=[(dir,'tank')]
                elif cats_found[0][1]=='tee' and cats_found[1][1]=='dress':
                    cats_found=[(dir,'dress')]
                elif cats_found[0][1]=='henley' and cats_found[1][1]=='tank':
                    cats_found=[(dir,'tank')]
                elif cats_found[0][1]=='tank' and cats_found[1][1]=='henley':
                    cats_found=[(dir,'tank')]
                elif cats_found[0][1]=='sweater' and cats_found[1][1]=='poncho':
                    cats_found=[(dir,'sweater')]
                elif cats_found[0][1]=='tank' and cats_found[1][1]=='tee':
                    cats_found=[(dir,'tank')]  #this could o either way
                elif cats_found[0][1]=='tee' and cats_found[1][1]=='jacket':
                    cats_found=[(dir,'jacket')]
                elif cats_found[0][1]=='sweater' and cats_found[1][1]=='henley':
                    cats_found=[(dir,'sweater')]
                elif cats_found[0][1]=='jeans' and cats_found[1][1]=='tee':
                    cats_found=[(dir,'jeans')] #velveteen jeans
                elif cats_found[0][1]=='tee' and cats_found[1][1]=='henley':
                    cats_found=[(dir,'henley')]  #Contrast_Trim_Henley_Tee
                elif cats_found[0][1]=='dress' and cats_found[1][1]=='tee':
                    cats_found=[(dir,'dress')]  #Jersey_Knit_Trapeze_Dress
                elif cats_found[0][1]=='sweatshirt' and cats_found[1][1]=='jacket':
                    cats_found=[(dir,'jacket')]
                elif cats_found[0][1]=='tee' and cats_found[1][1]=='jumpsuit':
                    cats_found=[(dir,'jumpsuit')] #Flounced_Sateen_Romper
                elif cats_found[0][1]=='cardigan' and cats_found[1][1]=='robe':
                    cats_found=[(dir,'cardigan')]  #free_spirit_kimono_cardigan -  toss out noncardigan stuf
                elif cats_found[0][1]=='coat' and cats_found[1][1]=='jacket':
                    cats_found=[(dir,'coat')]
                elif cats_found[0][1]=='hoodie' and cats_found[1][1]=='jacket':
                    cats_found=[(dir,'jacket')]
                elif cats_found[0][1]=='hoodie' and cats_found[1][1]=='henley':
                    cats_found=[(dir,'hoodie')]
                elif cats_found[0][1]=='sweater' and cats_found[1][1]=='dress':
                    cats_found=[(dir,'dress')] #turtleneck bodycon dress
                elif cats_found[0][1]=='sweater' and cats_found[1][1]=='hoodie':
                    cats_found=[(dir,'sweater')]
                elif cats_found[0][1]=='jeans' and cats_found[1][1]=='shorts':
                    cats_found=[(dir,'shorts')]
                elif cats_found[0][1]=='legging' and cats_found[1][1]=='tee':
                    cats_found=[(dir,'legging')] #velveteen leggings
                elif cats_found[0][1]=='sweater' and cats_found[1][1]=='dress':
                    cats_found=[(dir,'dress')]  #baroque floral sweater dress, marled_knit_sweaterr_dress
                elif cats_found[0][1]=='blouse' and cats_found[1][1]=='robe':
                    cats_found=[(dir,'blouse')] #kimono-sleeved-paisley-blouse
                elif cats_found[0][1]=='tee' and cats_found[1][1]=='robe':
                    cats_found=[(dir,'robe')]
                elif cats_found[0][1]=='jumpsuit' and cats_found[1][1]=='pants':
                    cats_found=[(dir,'jumpsuit')]
                elif cats_found[0][1]=='button-down' and cats_found[1][1]=='skirt':
                    cats_found=[(dir,'skirt')]
                elif cats_found[0][1]=='dress' and cats_found[1][1]=='henley':
                    cats_found=[(dir,'dress')]
                elif cats_found[0][1]=='tee' and cats_found[1][1]=='blouse':
                    cats_found=[(dir,'blouse')]
                elif cats_found[0][1]=='blouse' and cats_found[1][1]=='henley':
                    cats_found=[(dir,'blouse')]
                elif cats_found[0][1]=='poncho' and cats_found[1][1]=='dress':
                    cats_found=[(dir,'dress')]
                elif cats_found[0][1]=='tee' and cats_found[1][1]=='skirt':
                    cats_found=[(dir,'skirt')]
                elif cats_found[0][1]=='shorts' and cats_found[1][1]=='tee':
                    cats_found=[(dir,'shorts')]
                elif cats_found[0][1]=='cardigan' and cats_found[1][1]=='tee':
                    cats_found=[(dir,'cardigan')]

                elif cats_found[0][1]==None and cats_found[1][1] != None:
                    cats_found=[(dir,cats_found[1][1])]
                elif cats_found[0][1]!=None and cats_found[1][1] == None:
                    cats_found=[(dir,cats_found[1][0])]

                if len(cats_found)==1:
                    print('final disposition:'+str(cats_found))
                else:
                    print('NONFINAL disposition:'+str(cats_found))

        elif len(cats_found)==0:
            pass
#            print('dir {} no matching cats'.format(dir))
        elif len(cats_found)>2:
            print('3 or more! dir {} matching cats:{}'.format(dir,cats_found))
            if cats_found[1][1]=='sweater' and cats_found[2][1] == 'dress':
                cats_found=[(dir,'sweater')]
            elif cats_found[1][1]=='dress' and cats_found[2][1] == 'tee':
                cats_found=[(dir,'dress')]
            elif cats_found[1][1]=='button-down' and cats_found[2][1] == 'dress':
                cats_found=[(dir,None)]  #plaid flannel shirt dres s-half of these lookl like shirts and half dresses so aus
            elif cats_found[0][1]=='dress' and cats_found[1][1] == 'tee':
                cats_found=[(dir,'dress')]
            elif cats_found[1][1]=='dress' and cats_found[2][1] == 'henley':
                cats_found=[(dir,'dress')]
            elif cats_found[0][1]==cats_found[1][1]and cats_found[1][1]==cats_found[2][1]:
                cats_found=[(dir,cats_found[0][1])]
            elif cats_found[0][1]=='button-down' and cats_found[1][1] == 'dress':
                cats_found=[(dir,'dress')]

            if len(cats_found)==1:
                logging.info('final disposition:'+str(cats_found))
            else:
                logging.info('NONFINAL disposition:'+str(cats_found))

        if len(cats_found)==1:
            logging.info('unambiguous! '+str(cats_found))
            if cats_found[0][1] == None :
                logging.warning('got a None in dep_fashion_to_tg {}'.format(cats_found))
            all_cats.append(cats_found[0])  #add unambiguous cat to list
        else:
            logging.debug('AMBIGUOUS!!:'+str(cats_found))
                  #ambiguous (more thn one cat still) so dont add to list
    print('tot length '+str(len(all_cats)))
    return all_cats

def write_deepfashion_hydra_map_to_file():
    '''
    write the map between deepfashion folders and hydra cats to a file since this is needed for putting into db
    (cant do this on azure machines as they dont have db access from with docker , dont ask)
    :return:
    '''
    mapping = deepfashion_to_tg_hydra()
    mapfile = '/data/jeremy/image_dbs/labels/deepfashion_to_hydra_map.txt'
    Utils.ensure_file(mapfile)
    with open(mapfile,'w') as fp:
        json.dump(mapping,fp)

def generate_deep_fashion_hydra_labelfiles(folderpath='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/img',labelfile_dir='/data/jeremy/image_dbs/labels/hydra'):
    '''
    generate label file (lines like: /path/to/file.jpg class_no )
    using the deep fashion cats (from folder names) put into the hydra_cats lists
    does not overwrite the label files so delete if necessary before running
    :param folderpath:
    :param labelfile_dir:
    :return:
    '''
    dirs_and_cats = deepfashion_to_tg_hydra(folderpath=folderpath)

    pops ={}
    print('len dirs_and_cats:'+str(len(dirs_and_cats)))
    for dc in dirs_and_cats:
        print('dir,cat:')+str(dc)
        cat = dc[1]
        if cat in pops:
            pops[cat]+=1
        else:
            pops[cat]=1
    print pops
    raw_input('ret to cont')
    overall_populations = [[] for dummy in range(len(constants.hydra_cats))]
    for i in range(len(constants.hydra_cats)):   #iterate over category lists - whole_body, upper_cover etc
        catlist = constants.hydra_cats[i]
        labelfile_name = os.path.join(labelfile_dir,constants.hydra_cat_listlabels[i]+'_positive_labels.txt')
        Utils.ensure_file(labelfile_name)
        positives=[[] for dummy in range(len(catlist))]  #keep these positives for use as negatives against other cats
        #this will take some thinking since the positives can contain multiple cats...maybe need the multilabel db
        #just did this using the fully-labelled tamara-berg filipino images as I couldnt figure any other way to do it
        populations = [0 for dummy in range(len(catlist))] #
        print('doing categories in '+str(constants.hydra_cat_listlabels[i])+' cats:'+str(catlist))
        raw_input('ret to cont')
        with open(labelfile_name,'wa') as fp:
            for j in range(len(catlist)):   #iterate over indiv cats in catlist except for first, e.g. whole_body=[None, 'dress','suit',etc]
                cat = catlist[j]
                cat_index = j  #the 0th cat is always None e.g. [None,'dress','suit',etc]
                print('doing cat {} with index {} '.format(cat,cat_index))
                if cat is None:
                    continue
                for dir_cat_tuple in dirs_and_cats:
                    dir = dir_cat_tuple[0]
                    dircat = dir_cat_tuple[1]
                    if dircat==cat:
                        print('directory {} used for category {} index {}'.format(dir,dircat,cat_index))
                        full_path = os.path.join(folderpath,dir)
                        files = os.listdir(full_path)

                        positives[j].append(files)
                        for file in files:
                            file_path = os.path.join(full_path,file)
                            fp.write(file_path+'\t'+str(cat_index)+'\n')
                            logging.debug('wrote "{} {}" for file {} cat {}'.format(file_path,cat_index,file,cat_index))  #add no-cr
                            populations[cat_index]+=1
                        #raw_input('ret to cont')
        overall_populations[i] = populations
        print('populations of {} ({}) are {}'.format(constants.hydra_cat_listlabels[i],constants.hydra_cats[i],populations))

    #do negatives using positives of everythin else
    #possibly skew this towards hardest-to-differentiate (closest) cats e.g. more dresses as negs for skirts and vice versa

def deep_fashion_multiple_cat_labels(folderpath='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/img',
                                     labelfile_name='/data/jeremy/image_dbs/labels/hydra/skirt_v_dress_df.txt',cats=['skirt','dress'],cat_indices=[0,1]):
    for cat,ind in zip(cats,cat_indices):
        deep_fashion_single_cat_labels(folderpath=folderpath,labelfile_name=labelfile_name,cat=cat,cat_index=ind)
        raw_input('ret to cont')

def deep_fashion_single_cat_labels(folderpath='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/img',labelfile_name='/data/jeremy/image_dbs/labels/hydra/dress',cat='dress',cat_index=1,lookfor='250x250'):
    '''
    generate label file (lines like: /path/to/file.jpg class_no )
    using given cat (from folder names) put into desired cat_index
    does not overwrite the label files so delete if necessary before running
    :param folderpath:
    :param labelfile_dir:
    :return:
    '''
    dirs_and_cats = deepfashion_to_tg_hydra(folderpath=folderpath)
    pops = 0
    if labelfile_name is None:
        labelfile_name = cat+'_positives.txt'
    print('len dirs_and_cats:'+str(len(dirs_and_cats))+' labelfile '+labelfile_name+' folderpath '+folderpath)
    Utils.ensure_file(labelfile_name)
    linelist = []
    for dc in dirs_and_cats:
        if dc[1] == cat:
            print('dir,cat:')+str(dc)
            full_path = os.path.join(folderpath,dc[0])
            files = os.listdir(full_path)
            files = [f for f in files if lookfor in f]
            for file in files:
                file_path = os.path.join(full_path,file)
                line = file_path+'\t'+str(cat_index)+'\n'
                linelist.append(line)
                if labelfile_name:
                    with open(labelfile_name,'a') as fp:
                        fp.write(line)
                logging.debug('line {} for file {} cat {}'.format(line,file,cat_index))  #add no-cr
                pops+=1
        #        raw_input('ret to cont')
        print('population of {} (label {}) is {}'.format(cat,cat_index,pops))
        return(linelist)

def copy_relevant_deep_fashion_dirs_for_yonatan_features(deep_fashion_path='/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/img'):
    '''
    copy dirs from deep fashion into yonatan's training dirs
    :param folderpath:
    :return:
    '''
    dirs = os.listdir(deep_fashion_path)
    overall_populations = [[] for dummy in range(len(constants.hydra_cats))]
    for kw,feature in config.FEATURES.iteritems():   #iterate over category lists - whole_body, upper_cover etc
        print('kw {} feature {}'.format(kw,feature))

        path_to_images = feature['path_to_images']
        labels = feature['labels']
        find_labels = feature['labels']
        if kw == 'length':
            find_labels = {'mini':0,'midi':1,'maxi':2}
        elif kw == 'collar':
            find_labels= {
            'crew_neck': 0,
            'scoop_neck': 1,
            'v_neck': 2,
            'deep_v_neck': 3,
            'henley': 4,
            'polo': 5,
            'tie_neck': 6,
            'turtleneck': 7,
            'hoodie': 8,
            'strapless': 9
            }
        populations = {label:0 for label in find_labels} #
        print('doing categories: '+str(find_labels))
        raw_input('ret to cont')
        for label in find_labels:   #iterate over indiv cats in catlist except for first, e.g. whole_body=[None, 'dress','suit',etc]
            cat = label.lower()
            print('doing label {} '.format(cat))
            for dir in dirs:
                if cat.lower() in dir.lower():
                    orig_path = os.path.join(deep_fashion_path,dir)
                    label_index = find_labels[label]
                    found = False
                    orig = label
                    for orig_label in labels:
                        if labels[orig_label] == label_index:
                            orig = orig_label
                            found = True
                    if not found:
                        print('didnt find corresponding label for label_index {} label {}'.format(label_index,label))
                    dest_path = os.path.join(path_to_images,orig)
                    dest_path = os.path.join(dest_path,'deepfashion_'+dir)
                    print('directory {} used for category {}\ncopy  to {}'.format(orig_path,cat,dest_path))
                    Utils.ensure_dir(dest_path)
                    files = os.listdir(orig_path)
                    for file in files:
                        full_orig_path = os.path.join(orig_path,file)
                        full_dest_path = os.path.join(dest_path,file)
                        logging.debug('copy "{} to {}" for cat {}\n'.format(full_orig_path,full_dest_path,cat))  #add no-cr
                        copyfile(full_orig_path, full_dest_path)
                        populations[cat]+=1
                    #raw_input('ret to cont')
            raw_input('ret to cont')
        print('populations: {} '.format(populations))

def negatives_for_hydra(web_prefix='https://tg-training.storage.googleapis.com',
                        img_dir_prefix='/data/jeremy/image_dbs/tamara_berg_street_to_shop/photos',
                        neg_dir='/data/jeremy/image_dbs/labels/hydra'):
    '''
    Create negatives for the hydra cats using the multilabelled images from tamara berg
    (labels in constants.web_tool_categories_v2, data in db.training_images)
    make one negatives file for each outer hydra cat
    SPECIAL CASES
    there are no 'undies' (bra/panties/babydolls) in the multilabelled data
    negatives for those can just be everything in ml data
    multilabel footwear - shouldnt be used as neg for the hydra footwear (boots/shoes/sandals)
    multilabel top - dont use as neg for hydra 'shirt'
    :return:
    '''
    toplevel_cats = constants.hydra_cats
    db = constants.db
    cursor = db.training_images.find()
    n_done = cursor.count()
    print(str(n_done)+' docs done')
    #loop over docs in db
    for doc_count in range(n_done):
        document = cursor.next()
        if not 'already_seen_image_level' in document:
            print('no votes for this doc')
            continue
        if document['already_seen_image_level']<2:
            print('not enough votes for this doc')
            continue
        url = document['url']
        full_path  = url.replace(web_prefix,img_dir_prefix)
        full_path  = full_path.replace('tamara_berg_street2shop_dataset/images/','')


        if not os.path.exists(full_path):
            print('WARNING file '+full_path+' does not exist, but not skipping')
            #continue
        items_list = document['items']
        if items_list is None:
            print('no items in doc')
            continue
        logging.debug('items:'+str(items_list))
        votelist = [0]*len(constants.web_tool_categories_v2)
        #loop over items in doc
        for item in items_list:
            cat = item['category']
            if cat in constants.web_tool_categories_v2:
                index = constants.web_tool_categories_v2.index(cat)
            elif cat in constants.tamara_berg_to_web_tool_dict:
                print('old cat being translated')
                cat = constants.tamara_berg_to_web_tool_dict[cat]
                index = constants.web_tool_categories.index(cat)
            else:
                print('unrecognized cat')
                continue
            votelist[index] += 1
           # print('item:'+str(cat) +' votes:'+str(votelist[index]))

        print('doc '+str(doc_count)+ ' votes:'+str(votelist))
        nonzero_items={}
        for dummy in range(len(votelist)):
            if votelist[dummy]>0:
                nonzero_items[constants.web_tool_categories_v2[dummy]]=votelist[dummy]
        print('nonzero items:'+str(nonzero_items))
        #loop over possible roles as negative
        cats_i = 0
        for cat_list in constants.hydra_cats:
            logging.debug('trying catlist {}'.format(cat_list))
            useful_as_negative = True
            for cat in cat_list: #these are the  individual hydra cats
                if cat in constants.web_tool_categories_v2:
                    cat_index = constants.web_tool_categories_v2.index(cat)
                    if votelist[cat_index]!=0:
                        useful_as_negative = False
                        print('not useful due to {} with {} votes'.format(cat,votelist[cat_index]))
                        break
                    else:
                        logging.debug('possibly useful even due to {} with {} votes'.format(cat,votelist[cat_index]))
                else:
                # special cases where multilabel tag doesnt match hydra tag
                    #1. footwear
                    logging.debug('did not find {} in web_tool_cats_v2'.format(cat))
                    if cat in [ 'boots','shoes','sandals']:
                        cat_index = constants.web_tool_categories_v2.index('footwear')
                        if votelist[cat_index] != 0:
                            useful_as_negative = False
                            print('footwear:not useful due to footwear with {} votes'.format(votelist[cat_index]))
                            break
                        else:
                            logging.debug('footwear:possibly useful even due to {} with {} votes'.format(cat,votelist[cat_index]))
                    #2 top
                    elif cat in ['tee','button-down','blouse','polo','henley','tube','tank']:
                        cat_index = constants.web_tool_categories_v2.index('top')
                        if votelist[cat_index] != 0:
                            useful_as_negative = False
                            print('shirt:not useful due to top with {} votes'.format(votelist[cat_index]))
                            break
                        else:
                            logging.debug('shirt:possibly useful even due to {} with {} votes'.format(cat,votelist[cat_index]))

            if useful_as_negative:
                print('useful as negative for '+str(cat_list))
                filename = os.path.join(neg_dir,constants.hydra_cat_listlabels[cats_i]+'_negatives.txt')
                with open(filename,'a') as fp:
                    fp.write(full_path+'\t'+ str(0)+'\n')
                    fp.close()
         #       print('wrote {} to {}'.format(full_path+'\t'+str(0),filename))
            else:
                logging.debug('not useful as negative for '+str(cat_list))
            cats_i += 1
        #raw_input('ret to cont')




if __name__ == "__main__": #

    dir = '/home/jeremy/projects/core/'
    iamge_dir = 'images'
    annotation_dir='images'
    from trendi.downloaders import read_various_training_formats
    read_various_training_formats.inspect_yolo_annotations(dir='/home/jeremy/projects/core/',
                             yolo_annotation_folder='images',img_folder='images',manual_verification=False,
                             annotation_filter='_yololabels.txt')


#    write_cats_from_db_to_textfile()
#    split_to_trainfile_and_testfile()
#    inspect_textfile()

#test_u21_256x256_no_aug
#    dir_to_file_singlelabel(dir,classindex,labelfile,outfile=None,filter='.jpg'):
#    balance_cats(f)
#    outfilename = f.replace('.txt','')+'_balanced.txt'
#    split_to_trainfile_and_testfile(outfilename)


    '''x = ['bag_filipino_labels.txt',
         'belt_filipino_labels.txt',
         'bracelet_filipino_labels.txt',
         'cardigan_filipino_labels.txt',
         'coat_filipino_labels.txt',
         'dress_filipino_labels.txt',
         'earrings_filipino_labels.txt',
         'eyewear_filipino_labels.txt',
         'footwear_filipino_labels.txt',
         'hat_filipino_labels.txt',
         'jacket_filipino_labels.txt',
         'jeans_filipino_labels.txt',
         'necklace_filipino_labels.txt',
         'overalls_filipino_labels.txt',
         'pants_filipino_labels.txt',
         'scarf_filipino_labels.txt',
         'shorts_filipino_labels.txt',
         'skirt_filipino_labels.txt',
         'stocking_filipino_labels.txt',
         'suit_filipino_labels.txt',
         'sweater_filipino_labels.txt',
         'sweatshirt_filipino_labels.txt',
         'top_filipino_labels.txt',
         'watch_filipino_labels.txt',
         'womens_swimwear_bikini_filipino_labels.txt',
         'womens_swimwear_nonbikini_filipino_labels.txt']
    dir = '/data/jeremy/image_dbs/tamara_berg_street_to_shop/todo/'
    x = [os.path.join(dir,f) for f in os.listdir(dir) if '.txt' in f]
    x.sort()
    for f in x:
        balance_cats(f)
        outfilename = f.replace('.txt','')+'_balanced.txt'
        split_to_trainfile_and_testfile(outfilename)
'''
## change from photos to photos_250x250:
#sed s'/photos/photos_250x250/' bag_filipino_labels_balanced.txt > bag_filipino_labels_250x250.txt

    if(0):
        dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/'
        textfile_for_pixlevel(imagesdir=dir+'images/train_u21_256x256_no_aug',labelsdir=dir+'labels_256x256',outfilename=dir+'images_and_labelsfile_train.txt')
    #    split_to_trainfile_and_testfile(dir+'images_and_labelsfile.txt')
    #    inspect_pixlevel_textfile(dir+'images_and_labelsfile_train.txt')

        textfile_for_pixlevel(imagesdir=dir+'images/test_u21_256x256_no_aug',labelsdir=dir+'labels_256x256',outfilename=dir+'images_and_labelsfile_test.txt')
    #    split_to_trainfile_and_testfile(dir+'images_and_labelsfile.txt')
        inspect_pixlevel_textfile(dir+'images_and_labelsfile_test.txt')

#    deepfashion_to_tg_hydra()
#    generate_deep_fashion_hydra_labelfiles()
#    negatives_for_hydra()

    binary_pos_and_neg_deepfashion_onecat('dress')

        #useful script - change all photos to photos_250x250
#!/usr/bin/env bash
#echo $1
#name=$(echo $1|sed 's/.txt/_250x250.txt/')
#echo $name
#sed 's/photos/photos_250x250/' $1 > $name

#use with
#    for f in *.txt; do ./doit.sh $f; done
