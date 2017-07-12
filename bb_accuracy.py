__author__ = 'jeremy'

import msgpack
import requests

from trendi import constants
from trendi import Utils
from trendi.downloaders import read_various_training_formats

def multilabel_infer_one(url):
    image_mean = np.array([104.0,117.0,123.0])
    input_scale = None
    channel_swap = [2, 1, 0]
    raw_scale = 255.0
    print('loading caffemodel for neurodoll (single class layers)')

    start_time = time.time()
    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)   #
    in_ = np.array(in_, dtype=np.float32)   #.astype(float)
    if len(in_.shape) != 3:  #h x w x channels, will be 2 if only h x w
        print('got 1-chan image, turning into 3 channel')
        #DEBUG THIS , ORDER MAY BE WRONG [what order? what was i thinking???]
        in_ = np.array([in_,in_,in_])
    elif in_.shape[2] != 3:  #for rgb/bgr, some imgages have 4 chan for alpha i guess
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR
#    cv2.imshow('test',in_)
    in_ -= np.array((104,116,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = net.blobs['siggy'].data[0][category_index] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out shape {}'.format(min,max,out.shape))
    out = out*255
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out after scaling  {}'.format(min,max,out.shape))
    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
#    outname = os.path.basename(imagename)
#    outname = outname.split('.jpg')[0]+'.bmp'
#    outname = os.path.join(out_dir,outname)
#    print('outname:'+outname)
#    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('infer_one elapsed time:'+str(elapsed_time))
 #   cv2.imshow('out',out.astype(np.uint8))
 #   cv2.waitKey(0)
    return out.astype(np.uint8)

def test_yolo(testfilelist='/data/jeremy/image_dbs/hls/annotations',filter='.txt'):
    '''
    WIP
    run yolo on a dir having gt from kyle or elsewhere, get yolo  and compare
    :param dir:
    :return:
    '''
    with open(testfilelist,'r') as fp:
        lines = fp.readlines()
    for line in lines:
        img_file =
    for gt_file in gts:
        yolodict = read_various_training_formats.kyle_dicts_to_yolo(gt_file)
        apidict = read_various_training_formats.yolo_to_tgdict(yolodict)

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

####WIP #####

def test_multilabel_output_on_testset(testfile,outdir='./'):
    '''
    this takes an images+multilabels file and determines accuracy
    :param testfile: images+labels n testset
    :return:acc/precision/recall
    '''
    #get list of images, labels from file
    img_files=[]
    label_vecs=[]
    with open(testfile,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            imgfilename = line.split()[0]
            try:
                vals = line.split()[1:]
                n_labels = len(vals)
                label_vec = [int(i) for i in vals]
            except:
                logging.debug('got something that coulndt be turned into a string in the following line from file '+testfile)
                logging.debug(line)
                logging.debug('error:'+str(sys.exc_info()[0])+' , skipping line')
                continue
            label_vec = np.array(label_vec)
            n_labels = len(label_vec)
            if n_labels == 1:
        #                  print('length 1 label')
                label_vec = label_vec[0]
            img_files.append(imgfilename)
            label_vecs.append(label_vec)

    estimates=[]
    for imgfile in img_files:
        img_arr = cv2.imread(imgfile)
        if img_arr is None:
            logging.info('could not read '+str(imgfile))
            estimates.append(None)
            continue
        ml_output=get_multilabel_output_using_nfc(img_arr)
        print('ml output:'+str(ml_output))
        mlfilename = os.path.basename(imgfile).replace('.txt','_mloutput.txt')
        mlfilename = os.path.join(outdir,mlfilename)
        with(open(mlfilename,'a')) as fp:
            fp.write(ml_output)
        fp.close()
        estimates.append(ml_output)
    print('checking accuracy')
    check_acc_nonet(label_vecs,estimates)

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

def open_html(modelname,dir=None,solverproto='',caffemodel='',classlabels = constants.web_tool_categories,name=None):
    model_base = os.path.basename(modelname)
    if dir is not None:
        Utils.ensure_dir(dir)
        htmlname = os.path.join(dir,model_base+'results.html')
    else:
        htmlname = os.path.join(model_base,'results.html')
    if name is not None:
        htmlname = name
    with open(htmlname,'a') as g:
        g.write('<!DOCTYPE html>')
        g.write('<html>')
        g.write('<head>')
        g.write('<title>')
        dt=datetime.datetime.today()
        g.write(model_base+' '+dt.isoformat())
        g.write('</title>')
        g.write('solver:'+solverproto+'\n'+'<br>')
        g.write('model:'+caffemodel+'\n'+'<br>')
        g.write('</head>')
#        g.write('categories: '+str(constants.web_tool_categories)+'<br>'+'\n')
        g.write('<table style=\"width:100%\">\n')
        g.write('<tr>\n')
        g.write('<th>')
        g.write('metric')
        g.write('</th>\n')
        g.write('<th>')
        g.write('fw avg.')
        g.write('</th>\n')
        if classlabels:
            for i in range(len(classlabels)):
                g.write('<th>')
                g.write(classlabels[i])
                g.write('</th>\n')
        g.write('</tr>\n')
#        g.write('</table><br>')

def close_html(model_base,dir=None):
    if dir is None:
        protoname = solverproto.replace('.prototxt','')
        dir = 'multilabel_results-'+protoname+'_'+model_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    htmlname = os.path.join(dir,model_base+'results.html')
    with open(htmlname,'a') as g:
        g.write('</table><br>')
        plotfilename = 'multilabel_results'+model_base+'.png'

        g.write('<a href=\"'+plotfilename+'\">plot<img src = \"'+plotfilename+'\" style=\"width:300px\"></a>')
        g.write('</html>')

def summary_html(dir):
    htmlfiles = [f for f in os.listdir(dir) if '.html' in f]
    htmlname = os.path.join(dir,'summary.html')
    with open(htmlname,'w') as g:
        g.write('<!DOCTYPE html><br>\n')
        g.write('<html><br>\n')
        g.write('<head><br>\n')
        g.write('<title><br>\n')
        g.write('multilabel accuracy/precision/recall results')
        g.write('</title><br>\n')
        g.write('</head>\n')
        g.write('<br>\n')
        for h in htmlfiles:
            g.write('<a href=\"'+ h+'\"> '+h+'</a><br>\n')
        g.write('</html>')

#        g.write('categories: '+str(constants.web_tool_categories)+'<br>'+'\n')


def write_html(p,r,a,n,threshold,modelname,positives=False,dir=None,name=None,classlabels=None):
    model_base = os.path.basename(modelname)

    if dir is not None:
        Utils.ensure_dir(dir)
        htmlname = os.path.join(dir,model_base+'results.html')
    else:
        htmlname = os.path.join(model_base,'results.html')
    if name is not None:
        htmlname = name

    open_html(model_base,dir=dir,classlabels=classlabels)

    with open(htmlname,'a') as g:
        fwavp = 0
        fwavr = 0
        fwava = 0
        n_p=0
        n_r=0
        n_a=0
        fwavn = 0
        n_sum = 0
        #calculate frequency-weighted averages
        for i in range(len(p)):
            if not np.isnan(p[i]):
                fwavp = fwavp + p[i]*n[i]
                n_p=n_p+n[i]
            if not np.isnan(r[i]):
                fwavr = fwavr + r[i]*n[i]
                n_r=n_r+n[i]
            if not np.isnan(a[i]):
                fwava = fwava + a[i]*n[i]
                n_a=n_a+n[i]
            n_sum=n_sum+n[i]
        print('n sum {} fwavp {} fwavr {} fwava {} before division np {} nr {} na {} '.format(n_sum,fwavp,fwavr,fwava,n_p,n_r,n_a))
        fwavp = fwavp/float(n_p)
        fwavr = fwavr/float(n_r)
        fwava = fwava/float(n_a)
        fwavn = n_sum/float(len(p))

        print('frequency weighted averages p {} r {} acc {} n {}'.format(fwavp,fwavr,fwava,fwavn))
        g.write('frequency weighted averages p {} r {} acc {} n {}'.format(round(fwavp,3),round(fwavr,3),round(fwava,3),round(fwavn,3)))
    #write line with n_positives
        if(positives):
            g.write('<tr>\n')
            g.write('<td>')
            g.write('n_positives')
            g.write('</td>\n')
            g.write('<td>')
            g.write(str(round(fwavn,3)))
            g.write('</td>\n')
            for i in range(len(p)):
                g.write('<td>')
                g.write(str(int(n[i])))
                g.write('</td>\n')
            g.write('</tr>\n<br>\n')

    #write line with threshold
 #       g.write('<table style=\"width:100%\">\n')
        g.write('<b>')
        g.write('<tr>\n')
        g.write('<td>')
        g.write('threshold\n')
        g.write('</td>')
        g.write('<td>')
        g.write('')
        g.write('</td>\n')
        for i in range(len(p)):
            g.write('<td>')
            g.write(str(round(threshold,3)))
            g.write('</td>\n')
        g.write('</tr>\n')
        g.write('</b>')

    #write row with precision
        g.write('<tr>\n')
        g.write('<td>')
        g.write('precision')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(fwavp,3)))
        g.write('</td>\n')
        for i in range(len(p)):
            g.write('<td>')
            g.write(str(round(p[i],3)))
            g.write('</td>\n')
        g.write('</tr>\n')

    #write row with recall
        g.write('<tr>\n')
        g.write('<td>')
        g.write('recall')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(fwavr,3)))
        g.write('</td>\n')
        for i in range(len(p)):
            g.write('<td>')
            g.write(str(round(r[i],3)))
            g.write('</td>\n')
        g.write('</tr>\n')

    #write row with accuracy
        g.write('<tr>\n')
        g.write('<td>')
        g.write('accuracy')
        g.write('</td>\n')
        g.write('<td>')
        g.write(str(round(fwava,3)))
        g.write('</td>\n')
        for i in range(len(p)):
            g.write('<td>')
            g.write(str(round(a[i],3)))
            g.write('</td>\n')
        g.write('</tr>\n<br>\n')

        g.write('<tr><td><br/></td></tr>')  #blank row


#        g.write('threshold = '+str(t)+'\n')

def write_textfile(p,r,a,tp,tn,fp,fn,threshold,model_base,dir=None):
    if dir is None:
        protoname = solverproto.replace('.prototxt','')
        dir = 'multilabel_results-'+protoname+'_'+model_base.replace('.caffemodel','')
    Utils.ensure_dir(dir)
    fname = os.path.join(dir,model_base+'results.txt')
    with open(fname,'a') as f:
        f.write(model_base+' threshold = '+str(threshold)+'\n')
        f.write('solver:'+solverproto+'\n')
        f.write('model:'+caffemodel+'\n')
        f.write('categories: '+str(constants.web_tool_categories)+ '\n')
        f.write('precision\n')
        f.write(str(p)+'\n')
        f.write('recall\n')
        f.write(str(r)+'\n')
        f.write('accuracy\n')
        f.write(str(a)+'\n')
        f.write('true positives\n')
        f.write(str(tp)+'\n')
        f.write('true negatives\n')
        f.write(str(tn)+'\n')
        f.write('false positives\n')
        f.write(str(fp)+'\n')
        f.write('false negatives\n')
        f.write(str(fn)+'\n')
        f.close()


def get_traintest_from_proto(proto):
    print('looking for netname in '+proto)
    with open(proto,'r') as fp:
        train = None
        test = None
        traintest = None
        for line in fp:
            line = line.replace(' ','')  #line with spaces removed
            print('looking at line:'+line)
            if 'train_net:' in line and line[0] is not '#':
                train = line.replace('train_net:','').replace('"','')
                print('train:'+train)
            if 'test_net:' in line and line[0] is not '#':
                test = line.replace('test_net:','').replace('"','')
                print('test:'+test)
            if line[0:3] == 'net:'  and line[0] is not '#':
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




