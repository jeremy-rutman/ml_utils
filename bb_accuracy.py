__author__ = 'jeremy'

import msgpack
import requests

from trendi import constants
from trendi import Utils
from trendi.downloaders import read_various_training_formats

def compare_bb_dicts(gt_dict,guess_dict):
    '''
    given 2 dicts (in 'api form', see below of bbs - find bb in dict2 having most overlap for each bb in dict1 (assuming thats the gt)
    if it exists, check if categories match . return n_matching cats and avg. iou. iou=0 if no box is overlapping , and extra detections (false pos)
    also count for 0 iou

    :param dict1:ground truth in 'api form' {'data': [{'confidence': None, 'object': 'bag', 'bbox': [454, 306, 512, 360]},...,]}
    :param dict2:guess in 'api form'
    :return:  n_true_positive, n_false_neg, n_false_pos, iou
    '''
    gt_data=gt_dict['data']
    guess_data=guess_dict['data']
    true_pos = 0
    false_pos = 0
    false_neg = 0
    tot_objects = 0
    iou_tot = 0
    n = 0
    for gt_detection in gt_data:
        best_iou=0
        best_detection = None
        for guess_detection in guess_data:
            if 'already_matched' in guess_data:
                print('already matched guess {}'.format(guess_detection))
                continue
            iou = Utils.intersectionOverUnion(gt_detection['bbox'],guess_detection['bbox'])
            print('checking gt {} {} vs {} {}, iou {}'.format(gt_detection['bbox'],gt_detection['object'],
                                                                            guess_detection['bbox'],guess_detection['object'],iou))
            if iou>best_iou:
                best_iou = iou
                best_detection = guess_detection
        if best_detection is not None:
            best_detection['already_matched']=True #this gets put into original guess_detection
            gt_detection['already_matched']=True #this gets put into original guess_detection
            if best_detection['object'] == gt_detection['object'] and best_iou > 0:
                print('matching object {} in gt and guess, iou {}'.format(best_detection['object'],best_iou))
                true_pos += 1
            else:
                false_pos += 1
        tot_objects += 1
        iou_tot += best_iou
        n += 1
    #check for extra guess detections
    for guess_detection in guess_data:
        if not 'already_matched' in guess_detection:
            false_pos += 1
            n+=1
    for gt_detection in gt_data:
        if not 'already_matched' in gt_detection:
            false_neg += 1
            n+=1
    iou_avg = iou_tot/n
    return {'tp':true_pos,'tn':false_pos,'fn':false_neg,'iou_avg':iou_avg}

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

def bb_output_using_gunicorn(url_or_np_array):
    print('starting get_multilabel_output_using_nfc')
    multilabel_dict = nfc.pd(url_or_np_array, get_multilabel_results=True)
    logging.debug('get_multi_output:dict from falcon dict:'+str(multilabel_dict))
    if not multilabel_dict['success']:
        logging.warning('did not get nfc pd result succesfully')
        return
    multilabel_output = multilabel_dict['multilabel_output']
    logging.debug('multilabel output:'+str(multilabel_output))
    return multilabel_output #

def bb_output_yolo_using_api(url_or_np_array,CLASSIFIER_ADDRESS=constants.YOLO_HLS_CLASSIFIER_ADDRESS,roi=None):
    print('starting bb_output_api at addr '+str(CLASSIFIER_ADDRESS))
#    CLASSIFIER_ADDRESS =   # "http://13.82.136.127:8082/hls"
    print('using yolo api addr '+str(CLASSIFIER_ADDRESS))
    if isinstance(url_or_np_array,basestring): #got a url (or filename, but not dealing with that case)
        data = {"imageUrl": url_or_np_array}
        print('using imageUrl as data')
    else:
        img_arr = Utils.get_cv2_img_array(url_or_np_array)
        data = {"image": img_arr} #this was hitting 'cant serialize' error
        print('using imgage as data')
    if roi:
        print "Make sure roi is a list in this order [x1, y1, x2, y2]"
        data["roi"] = roi
    serialized_data = msgpack.dumps(data)
#    resp = requests.post(CLASSIFIER_ADDRESS, data=serialized_data)
    result = requests.get(CLASSIFIER_ADDRESS,params=data)
    if result.status_code is not 200:
       print("Code is not 200")
#     else:
#         for chunk in result.iter_content():
#             print(chunk)
# #            joke = requests.get(JOKE_URL).json()["value"]["joke"]

#    resp = requests.post(CLASSIFIER_ADDRESS, data=data)
    c=  result.content
    #content should be roughly in form
#    {"data":
    # [{"confidence": 0.366, "object": "car", "bbox": [394, 49, 486, 82]},
    # {"confidence": 0.2606, "object": "car", "bbox": [0, 116, 571, 462]}, ... ]}
    if not 'data' in c:
        print('didnt get data in result from {} on sendng {}'.format(CLASSIFIER_ADDRESS,data))
    return data
    # t = result.text
    # print('content {} text {}'.format(c,t))
