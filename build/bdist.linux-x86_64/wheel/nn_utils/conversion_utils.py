__author__ = 'jeremy'

import os
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import json

from trendi import constants
from trendi.utils import imutils
from trendi import Utils

def convert_labels_dir(indir,outdir,jpgdir=None,converter=constants.fashionista_aug_zerobased_to_pixlevel_categories_v2,
                      suffix_in='.png',suffix_out='_pixlevelv2.bmp',for_webtool=False,
                      inlabels=constants.fashionista_categories_augmented_zero_based,
                      outlabels=constants.pixlevel_categories_v2, save_legends=True):
    '''
    convert e..g from paperdoll to ultimate21 or pixlevel_categories_v2 .
    Optionally only convert R channel for use with webtool. Don't forget to convert back to all chans after done w webtool
    :param dir:
    :param converter:
    :param input_suffix:
    :param for_webtool:
    :return:
    '''
    Utils.ensure_dir(outdir)
    files = [os.path.join(indir,f) for f in os.listdir(indir) if suffix_in in f]
    print('STARTING CONVERT - converting '+str(len(files))+' files in '+indir)
    for f in files:
        print('')
        newname = os.path.join(outdir,os.path.basename(f))
        newname = newname.replace(suffix_in,suffix_out)
        print('converting {} to {} '.format(f,newname))
        converted_arr = convert_labels(f,converter=converter,for_webtool=for_webtool,inlabels=inlabels,outlabels=outlabels)
        cv2.imwrite(newname,converted_arr)
        #raw_input('ret to cont')
        if save_legends:
            if jpgdir is None:
                jpgdir=indir
            orig_imagename=os.path.basename(f).replace(suffix_in,'.jpg')
            orig_imagename=os.path.join(jpgdir,orig_imagename)
            print('saving legend using {} '.format(orig_imagename))

            imutils.show_mask_with_labels(converted_arr,outlabels,original_image=orig_imagename,save_images=True)



def convert_labels(filename_or_img_array,converter=constants.fashionista_aug_zerobased_to_pixlevel_categories_v2,
                      for_webtool=True,inlabels=constants.fashionista_categories_augmented_zero_based,
                      outlabels=constants.pixlevel_categories_v2):
    '''
    convert e..g from paperdoll to ultimate21 or pixlevel_categories_v2 .
    Optionally only convert R channel for use with webtool. Don't forget to convert back to all chans after done w webtool
    :param converter:
    :param input_suffix:
    :param for_webtool:
    :return:
    '''
    if isinstance(filename_or_img_array,basestring):
        img_arr = cv2.imread(filename_or_img_array)
        filename = filename_or_img_array
    else:
        img_arr = filename_or_img_array
        filename = None
    if img_arr is None:
        logging.debug('got null image in conversion_utils.convert_pd_output')
    h,w = img_arr.shape[0:2]
    out_arr = np.zeros((h,w,3),dtype=np.uint8)
    for u in np.unique(img_arr):
        logging.debug('in converter, u='+str(u)+'len='+str(len(converter)))
        if u+1>len(converter):
            print('index {} is past length {} of converter, forcing to 0'.format(u,len(converter)))
            newindex=0
        else:
            newindex= converter[u]
        if newindex==None:
            newindex=0
        try:
            print('converting {} {} to {} {}'.format(u,inlabels[u],newindex,outlabels[newindex]))
        except:
            logging.warning('looks like index {} is greater than inlabel array length {}!!!'.format(u,len(inlabels)))
        out_arr[img_arr==u] = newindex  #B it would seem this can be replaced by out_arr[:,:,:]=img_arr, maybe :: is used here
    if for_webtool:
        out_arr[:,:,0:2] = 0
    return out_arr

def count_values(mask,labels=None):
    image_size = mask.shape[0]*mask.shape[1]
    uniques = np.unique(mask)
    pixelcounts = {}
    if len(mask.shape) == 3:
        mask = mask[:,:,0]  #this should be chan 3 if its a webtool image
    for unique in uniques:
        pixelcount = len(mask[mask==unique])
        ratio = float(pixelcount)/image_size
        if labels is not None:
            print('class {} {} count {} ratio {}'.format(unique,labels[unique],pixelcount,ratio))
        else:
            print('class {} count {} ratio {}'.format(unique,pixelcount,ratio))
        pixelcounts[unique]=pixelcount
    return pixelcounts

def test_many_conversions():
    multilabel_to_ultimate21_conversion=constants.binary_classifier_categories_to_ultimate_21
    multilabel_labels=constants.binary_classifier_categories
    print('testing binary classifier to u21 cats')
    print('ml2u21 conversion:'+str(multilabel_to_ultimate21_conversion))
    print('ml labels:'+str(multilabel_labels))
    for i in range(len(multilabel_labels)):
        neurodoll_index = multilabel_to_ultimate21_conversion[i]
        #print('nd index:'+str(neurodoll_index))
        if neurodoll_index is None:
            print('no mapping from index {} (label {}) to neurodoll'.format(i,multilabel_labels[i]))
            continue
        print('index {} webtoollabel {} newindex {} neurodoll_label {}'.format(i,
            multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index]))

    multilabel_to_ultimate21_conversion=constants.web_tool_categories_v1_to_ultimate_21
    multilabel_labels=constants.web_tool_categories
    print('testing webtool v2 to u21 cats')
    print('ml2u21 conversion:'+str(multilabel_to_ultimate21_conversion))
    print('ml labels:'+str(multilabel_labels))
    for i in range(len(multilabel_labels)):
        neurodoll_index = multilabel_to_ultimate21_conversion[i]
        if neurodoll_index is None:
            print('no mapping from index {} (label {}) to neurodoll'.format(i,multilabel_labels[i]))
            continue
        print('index {} webtoollabel {} newindex {} neurodoll_label {}'.format(i,
            multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index]))

    multilabel_to_ultimate21_conversion=constants.web_tool_categories_v2_to_ultimate_21
    multilabel_labels=constants.web_tool_categories_v2
    print('testing webtool v1 to u21 cats')
    print('ml2u21 conversion:'+str(multilabel_to_ultimate21_conversion))
    print('ml labels:'+str(multilabel_labels))
    for i in range(len(multilabel_labels)):
        neurodoll_index = multilabel_to_ultimate21_conversion[i]
        if neurodoll_index is None:
            print('no mapping from index {} (label {}) to neurodoll'.format(i,multilabel_labels[i]))
            continue
        print('index {} webtoollabel {} newindex {} neurodoll_label {}'.format(i,
            multilabel_labels[i],neurodoll_index,constants.ultimate_21[neurodoll_index]))

    converter=constants.fashionista_aug_zerobased_to_pixlevel_categories_v2
    orig_labels=constants.fashionista_categories_augmented_zero_based
    dest_labels=constants.pixlevel_categories_v2
    print('testing fashionista aug 0-based to pixlevel_v2 cats')
    for i in range(len(orig_labels)):
        dest_index = converter[i]
        if dest_index is None:
            print('no mapping from index {} (label {}) to dest'.format(i,orig_labels[i]))
            continue
        print('index {} origlabel {} newindex {} destlabel {}'.format(i,
            orig_labels[i],dest_index,dest_labels[dest_index]))

def test_convert(orig_labels,dest_labels,converter):

    print('testing conversion')
    for i in range(len(orig_labels)):
        dest_index = converter[i]
        if dest_index is None:
            print('no mapping from index {} (label {}) to dest'.format(i,orig_labels[i]))
            continue
        print('index {} origlabel {} newindex {} destlabel {}'.format(i,
            orig_labels[i],dest_index,dest_labels[dest_index]))

def gen_json(images_dir='data/pd_output',annotations_dir='data/pd_output',
             outfile = 'data/pd_output.json',labels=constants.pixlevel_categories_v2,mask_suffix='_pixv2_webtool.png',
             ignore_finished=True,finished_mask_suffix='_pixv2_webtool_finished_mask.png'):
    images = [os.path.join(images_dir,f) for f in os.listdir(images_dir) if '.jpg' in f and not 'legend' in f]
    the_dict = {'labels': labels, 'imageURLs':[], 'annotationURLs':[]}

    for f in images:
        print('looking at '+f)
        annotation_file = os.path.basename(f).replace('.jpg',mask_suffix)
        annotation_file = os.path.join(annotations_dir,annotation_file)
        if ignore_finished:
            maskname = annotation_file.replace(mask_suffix,finished_mask_suffix)
            #print('finished maskname:'+maskname)
            if os.path.isfile(maskname):
                print('mask '+maskname+' exists, skipping')
                continue
        if not os.path.isfile(annotation_file):
            print('could not find '+str(annotation_file))
            continue
        the_dict['imageURLs'].append(f)
        the_dict['annotationURLs'].append(annotation_file)
        print('added image '+f+' mask '+annotation_file)
    with open(outfile,'w') as fp:
        json.dump(the_dict,fp,indent=4)



if __name__ == "__main__":

#    gen_json()
    print('starting test')
    #test_convert(constants.ultimate_21,constants.pixlevel_categories_v3,constants.ultimate_21_to_pixlevel_v3)
    #test_convert(constants.fashionista_categories_augmented,constants.pixlevel_categories_v3,constants.fashionista_augmented_to_pixlevel_v3)
    test_convert(constants.fashionista_categories_augmented_zero_based,constants.pixlevel_categories_v4_for_web,constants.fashionista_aug_zerobased_to_pixlevel_categories_v4_for_web)
