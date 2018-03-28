__author__ = 'jeremy'

import os
import logging
logging.basicConfig(level=logging.INFO)  #debug is actually lower than info: critical/error/warning/info/debug
import random

class load_batch():
    '''
        needs to have self.images (np array) , self.labels , self.test.images, self.test.labels
        and to implement train.next_batch(50)
    '''
    def __init__(self,label_suffix='.json'):
        self.unified_label_and_data_file=True #alternative is one labelfile  per datafile
        self.label_type='single label' #or 'multilabel' or 'regression'
        self.trainfile = None #file containing list of training files
        self.testfile = None #file containing list of test files
        self.trainfiles = None #list of training data+label files (lines like trainfile1.txt  labelfile1.txt)
        self.testfiles = None #list of test data + label files
        self.label_dir ='/data/jeremy/image_dbs/variant/api' #assuming all files in one dir
        self.data_dir ='/data/jeremy/image_dbs/variant/api'
        self.file_counter = 0
        self.label_suffix = '.json'  #if no suffix, set to '' - so that ('' in 'string'==True)
        self.data_suffix = '.jpg' #if no suffix, set to '' - so that ('' in 'string'==True)
        if self.trainfile is None and self.testfile is None:
            self.find_train_and_testfiles()
        if self.trainfile is None and self.testfile is None:
            self.read_dirs()
        elif self.testfile is None:
            self.split_to_trainfile_and_testfile()
        #todo - implement the alternative
#
    def find_train_and_testfiles(self):
        trainfilename = os.path.join(self.label_dir,'train.txt')
        if not os.path.exists(trainfilename):
            self.read_dirs()
            self.split_to_trainfile_and_testfile()
            return
        testfilename = os.path.join(self.label_dir,'test.txt')
        if not os.path.exists(testfilename):
            self.split_to_trainfile_and_testfile()

    def read_dirs(self):
        self.datafile_labelfile_pairs=[]
        label_files = [os.path.join(self.label_dir,f) for f in os.listdir(self.label_dir) if self.label_suffix in f ]
        for l in label_files:
            datafile_name=os.path.basename(l).replace(self.label_suffix,self.data_suffix)
            datafile_name=os.path.join(self.data_dir_test,datafile_name)
            if os.path.exists(datafile_name):
                self.datafile_labelfile_pairs.append({'datafile':datafile_name,'labelfile':l})
            else:
                print('couldnt find datafile {} corresponding to labelfile {}'.format(datafile_name,l))

        # if self.label_dir_train==self.label_dir_test :
        #     self.split_to_test_and_train()
        #     print('')
        #     return

    def next_batch(self,n):
        self.batch=[]
        for filepair in self.trainfiles:
            datafile = filepair['data']
            labelfile = filepair['label']
            data=self.read_data(datafile)
            label=self.read_label(labelfile)
            if data is None or label is None:
                continue

    def read_data(self,datafile):
        data=cv2.imread(datafile)
        return(data)

    def read_label(self,labelfile):
        with open(labelfile,'r') as fp:
            line = fp.readlines()[0]

        vals = line.split()[1:]
        self.n_labels = len(vals)

        if self.regression:
            label_vec = [float(i) for i in vals]
        else:
            try:
                label_vec = [int(i) for i in vals]
            except:
                logging.debug('got something that coulndt be turned into a string in the following line from file ')
                logging.debug(line)
                logging.debug('error:'+str(sys.exc_info()[0])+' , skipping line')
                continue
        label_vec = np.array(label_vec)

        return(data)

    def split_to_trainfile_and_testfile(self,filename=None, fraction=0.05,shuffle=True):
        '''
        writes (destructively) files with _train.txt and _test.txt based on filename, with sizes determined by fraction
        :param filename: input catsfile
        :param fraction: ratio test:train
        :return:
        '''
        if filename is not None:
            with open(filename,'r') as fp:
                lines = fp.readlines()
                if lines == []:
                    logging.warning('nothing in {}'.format(filename))
                    return
                print('file {} has lines like {}'.format(filename,lines[0]))
            self.datafile_labelfile_pairs = None
            for line in lines:
                datafile,labelfile=line.split()
                self.datafile_labelfile_pairs.append({'datafile':datafile,'labelfile':labelfile})
        else:
            filename = os.path.join(self.label_dir,'train.txt')
        if shuffle:
            random.shuffle(self.datafile_labelfile_pairs)


        n_lines = len(self.datafile_labelfile_pairs)
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
