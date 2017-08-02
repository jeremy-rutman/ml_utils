__author__ = 'jeremy'
import caffe
import time
import os
import sys
import subprocess
import socket
import datetime
import numpy as np
from scipy.optimize import curve_fit


from trendi import Utils
from trendi import constants
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.classifier_stuff.caffe_nns import single_label_accuracy
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy
from trendi.classifier_stuff.caffe_nns import caffe_utils
import matplotlib as plt


def dosolve(weights,solverproto,testproto,type='single_label',steps_per_iter=1,n_iter=200,n_loops=200,n_tests=1000,
          cat=None,classlabels=None,baremetal_hostname='brainiK80a',solverstate=None,label_layer='label',estimate_layer='my_fc2'):

    if classlabels is None:
        classlabels=['not_'+cat,cat]
    caffe.set_device(int(sys.argv[1]))
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solverproto)
    if weights is not None:
        solver.net.copy_from(weights)
    if solverstate is not None:
        solver.restore(solverstate)   #see https://github.com/BVLC/caffe/issues/3651
        #No need to use solver.net.copy_from(). .caffemodel contains the weights. .solverstate contains the momentum vector.
    #Both are needed to restart training. If you restart training without momentum, the loss will spike up and it will take ~50k i
    #terations to recover. At test time you only need .caffemodel.
    training_net = solver.net
    solver.test_nets[0].share_with(solver.net)  #share train weight updates with testnet
    test_net = solver.test_nets[0] # more than one testnet is supported

    #get netname, train_test train/test
    net_name = caffe_utils.get_netname(testproto)
    tt = caffe_utils.get_traintest_from_proto(solverproto)
    print('netname {} train/test {}'.format(net_name,tt))

    docker_hostname = socket.gethostname()

    datestamp = datetime.datetime.strftime(datetime.datetime.now(), 'time%H.%M_%d-%m-%Y')
    prefix = baremetal_hostname+'_'+net_name+'_'+docker_hostname+'_'+datestamp


    #detailed_jsonfile = detailed_outputname[:-4]+'.json'
    if weights:
        weights_base = os.path.basename(weights)
    else:
        weights_base = '_noweights_'
    threshold = 0.5
    if net_name:
        outdir = type + '_' + prefix + '_' + weights_base.replace('.caffemodel','')
    else:
        outdir = type + '_' + prefix + '_' +testproto+'_'+weights_base.replace('.caffemodel','')
    outdir = outdir.replace('"','')  #remove quotes
    outdir = outdir.replace(' ','')  #remove spaces
    outdir = outdir.replace('\n','')  #remove newline
    outdir = outdir.replace('\r','')  #remove return
    outdir = './'+outdir

    #generate report filename, outdir to save everything (loss, html etc)
    if type == 'pixlevel':
        outname = os.path.join(outdir,outdir[2:]+'_netoutput.txt')  #TODO fix the shell script to not look for this, then it wont be needed
    if type == 'multilabel':
        outname = os.path.join(outdir,outdir[2:]+'_mlresults.html')
    if type == 'single_label':
        outdir = outdir + '_' + str(cat)
        outname = os.path.join(outdir,outdir[2:]+'_'+cat+'_slresults.txt')
    loss_outputname = os.path.join(outdir,outdir[2:]+'_loss.txt')
    print('outname:{}\n lossname {}\n outdir {}\n'.format(outname,loss_outputname,outdir))
    Utils.ensure_dir(outdir)
    time.sleep(0.1)
    Utils.ensure_file(loss_outputname)

    #copy training and test files to outdir
    if tt is not None:
        if len(tt) == 1:  #copy single traintest file to dir of info
            copycmd = 'cp '+tt[0] + ' ' + outdir
            subprocess.call(copycmd,shell=True)
        else:  #copy separate train and test files to dir of info
            copycmd = 'cp '+tt[0] + ' ' + outdir
            subprocess.call(copycmd,shell=True)
            copycmd = 'cp '+tt[1] + ' ' + outdir
            subprocess.call(copycmd,shell=True)
    #cpoy solverproto to results dir
    if solverproto is not None:
        copycmd = 'cp '+solverproto + ' ' + outdir
        subprocess.call(copycmd,shell=True)
    #copy test proto to results dir
    if testproto is not None:
        copycmd = 'cp '+testproto + ' ' + outdir
        subprocess.call(copycmd,shell=True)
    #copy this file too
    copycmd = 'cp solve_jr.py '  + outdir
    #if name o fthis file keeps changing we can use
    # os.path.realpath(__file__)  which gives name of currently running file
    subprocess.call(copycmd,shell=True)


    #copycmd = 'cp -r '+outdir + ' ' + host_dirname
    #copy to server
    scpcmd = 'rsync -avz '+outdir + ' root@104.155.22.95:/var/www/results/'+type+'/'

    #put in standard dir
    standard_dir = '/data/results/'+type+'/'
    Utils.ensure_dir(standard_dir)
    scpcmd2 = 'rsync -avz '+outdir + ' /data/results/'+type+'/'

    i = 0
    losses = []
    iters = []
#    loss_avg = [0]*n_iter
    loss_avg = np.zeros(n_iter)
#    accuracy_list = [0]*n_iter
    accuracy_list = np.zeros(n_iter)
    tot_iters = 0
    iter_list = []
    accuracy_list = []
    #instead of taking steps its also possible to do
    #solver.solve()

    if type == 'multilabel':
        multilabel_accuracy.open_html(weights, dir=outdir,solverproto=solverproto,caffemodel=weights,classlabels = constants.web_tool_categories_v2,name=outname)

    for _ in range(n_loops):
        for i in range(n_iter):
            solver.step(steps_per_iter)
    #        loss = solver.net.blobs['score'].data
            loss = solver.net.blobs['loss'].data
            loss_avg[i] = loss
            losses.append(loss)
            tot_iters = tot_iters + steps_per_iter
#            if type == 'single_label' or type == 'pixlevel': #test, may not work for pixlevel? #indeed does not work for pix
            if type == 'single_label' : #test, may not work for pixlevel? #indeed does not work for pix
                accuracy = solver.net.blobs['accuracy'].data
                accuracy_list[i] = accuracy
                print('iter '+str(i*steps_per_iter)+' loss:'+str(loss)+' acc:'+str(accuracy))
            else:
                print('iter '+str(i*steps_per_iter)+' loss:'+str(loss))

        iter_list.append(tot_iters)

        try:
            averaged_loss=np.average(loss_avg)
            s2 = '{}\t{}\n'.format(tot_iters,averaged_loss)
        except:
            print("something wierd with loss:"+str(loss_avg))
            s=0
            for i in loss_avg:
                print i
                s=s+i
            averaged_loss = s/len(loss_avg)
            print('avg:'+str(s)+' '+str(averaged_loss))

            s2 = '{}\t{}\n'.format(tot_iters,averaged_loss)

        #for test net:
    #    solver.test_nets[0].forward()  # test net (there can be more than one)
    #    progress_plot.lossplot(loss_outputname)  this hits tkinter problem
        if type == 'multilabel': #here accuracy is a list....jesus who wrote this
            precision,recall,accuracy,tp,tn,fp,fn = multilabel_accuracy.check_acc(test_net, num_samples=n_tests, threshold=0.5, gt_layer=label_layer,estimate_layer=estimate_layer)
            n_occurences = [tp[i]+fn[i] for i in range(len(tp))]
            multilabel_accuracy.write_html(precision,recall,accuracy,n_occurences,threshold,weights,positives=True,dir=outdir,name=outname,classlabels=classlabels)
            avg_accuracy = np.mean(accuracy)
            print('solve.py: loss {} p {} r {} a {} tp {} tn {} fp {} fn {}'.format(averaged_loss,precision,recall,accuracy,tp,tn,fp,fn))
            s2 = '{}\t{}\t{}\n'.format(tot_iters,averaged_loss,avg_accuracy)

        elif type == 'pixlevel':
                    # number of tests for pixlevel
            s = '#########\navg loss over last {} steps is {}'.format(n_iter*steps_per_iter,averaged_loss)
            print(s)
            # avg_accuracy = np.mean(accuracy)
            # print('accuracy mean {} std {}'.format(avg_accuracy,np.std(accuracy_list)))
            val = range(0,n_tests) #
            results_dict = jrinfer.seg_tests(solver,  val, output_layer=estimate_layer,gt_layer='label',outfilename=outname,save_dir=outdir,labels=classlabels)
#            results_dict = jrinfer.seg_tests(test_net,  val, output_layer=estimate_layer,gt_layer='label',outfilename=outname,save_dir=outdir,labels=classlabels)
                    #dont need to send test_net, the jrinfer already looks for test net part of solver
            overall_acc = results_dict['overall_acc']
            mean_acc = results_dict['mean_acc']
            mean_ion = results_dict['mean_iou']
            fwavacc = results_dict['fwavacc']
            s2 = '{}\t{}\t{}\n'.format(tot_iters,averaged_loss,overall_acc,mean_acc,mean_ion,fwavacc)

        elif type == 'single_label':
            averaged_acc = np.average(accuracy_list)
            accuracy_list.append(averaged_acc)
            s = 'avg tr loss over last {} steps is {}, acc:{} std {'.format(n_iter*steps_per_iter,averaged_loss,averaged_acc,np.std(accuracy_list))
            print(s)
#            print accuracy_list
            s2 = '{}\t{}\t{}\n'.format(tot_iters,averaged_loss,averaged_acc)

            acc = single_label_accuracy.single_label_acc(weights,testproto,net=test_net,label_layer='label',estimate_layer=estimate_layer,n_tests=n_tests,classlabels=classlabels,save_dir=outdir)
     #       test_net = solver.test_nets[0] # more than one testnet is supported
    #        testloss = test_net.blobs['loss'].data
            try:
                testloss = test_net.blobs['loss'].data
            except:
                print('no testloss available')
                testloss=0
            with open(loss_outputname,'a+') as f:
                f.write('test\t'+str(int(time.time()))+'\t'+str(tot_iters)+'\t'+str(testloss)+'\t'+str(acc)+'\n')
                f.close()
#            params,n_timeconstants = fit_points_exp(iter_list,accuracy_list)
#            print('fit: asymptote {} tau {} x0 {} t/tau {}'.format(params[0],params[1],params[2],n_timeconstants))
##            if n_timeconstants > 10 and tot_iters>10000:  #on a long tail
 #               return params,n_timeconstants

        with open(loss_outputname,'a+') as f:
            f.write(str(int(time.time()))+'\t'+s2)
            f.close()
    ##
    #   subprocess.call(copycmd,shell=True)
        subprocess.call(scpcmd,shell=True)
        subprocess.call(scpcmd2,shell=True)

def expfunc(x,asymptote,timeconst,x0):
    eps = 10**-5
    y = asymptote * (1-np.exp(-(x-x0)/(timeconst)))
    print('as {} tm {} x0 {}'.format(asymptote,timeconst,x0))
    return y
#    return a * np.exp(-b * x) + c
#a * np.exp(-b * x) + c

def fit_points_exp(xlist,ylist):
    p0 = {'asymptote':0.8,'timeconst':2000,'y0':0.7}
    p0 = [0.8,2000,0]
    popt,pcov = curve_fit(expfunc, xlist, ylist,p0=p0)
    print('popt {} pcov {}'.format(popt,pcov))
    n_timeconstants = xlist[-1]/popt[1]  #last time point / timeconst
    return(popt,n_timeconstants)

def test_fit():
#    x=np.linspace(0,10000,100)
#    y=expfunc(x,0.8,2000,100)
#    y = y + 0.2*np.random.rand(len(y))*y
    with open('/home/jeremy/projects/core/classifier_stuff/caffe_nns/loss3.txt','r') as fp:
        r = fp.readlines()
        x = [int(l.split()[1]) for l in r]
        y = [float(l.split()[-1]) for l in r ]
        x = np.array(x)
        y=np.array(y)
        print x,y
        popt,nt = fit_points_exp(x,y)
        y_est = expfunc(x,popt[0],popt[1],popt[2])
        n_timeconstants = x[-1]/popt[1]
        print('ntimesconsts {} nt {}'.format(n_timeconstants,nt))
        plt.plot(x,y,x,y_est)
        plt.show()
        print y_est


def solve_a_bunch():

    base_dir = '/home/jeremy/caffenets/binary/resnet101_dress_try1/'
    weights =  '/home/jeremy/caffenets/binary/ResNet-101-model.caffemodel'
    solverproto = base_dir + 'ResNet-101_solver.prototxt'
    testproto = base_dir + 'ResNet-101-train_test.prototxt'
    type='single_label'
    #type='multilabel'
    #type='pixlevel'
    steps_per_iter = 1
    n_iter = 200
    cat = "dress"
#    classlabels=['dress','not_dress']
    classlabels=constants.pixlevel_categories_v3
    n_tests = 2000
    n_loops = 2000000
    baremetal_hostname = 'k80b'
    label_layer='label'
    estimate_layer='fc4_0'
    dosolve(weights,solverproto,testproto,type=type,steps_per_iter=steps_per_iter,n_iter=n_iter,n_loops=n_loops,n_tests=n_tests,
          cat=cat,classlabels=classlabels,baremetal_hostname=baremetal_hostname,label_layer=label_layer,estimate_layer=estimate_layer)

if __name__ == "__main__":
###############
#vars to change
###############
#ResNet-101-deploy.prototxt  ResNet-101-train_test.prototxt  ResNet-101_solver.prototxt  snapshot  solve.py
    solverstate = None
    #base_dir = '/data/jeremy/caffenets/binary/resnet101_dress_try1/'
    base_dir = os.path.dirname(os.path.realpath(__file__))
    weights =  '/data/jeremy/caffenets/binary/ResNet-101-model.caffemodel'
    solverproto = os.path.join(base_dir,'ResNet-101_solver.prototxt')
    testproto = os.path.join(base_dir,'ResNet-101-train_test.prototxt')
    type='single_label'
    #type='multilabel'
    #type='pixlevel'
    steps_per_iter = 1
    n_iter = 200
    cat = "dress"
#    classlabels=['dress','not_dress']
    classlabels=constants.pixlevel_categories_v3
    n_tests = 2000
    n_loops = 2000000
    baremetal_hostname = 'k80b'
    label_layer='label'
    estimate_layer='fc2'
####################

    dosolve(weights,solverproto,testproto,type=type,steps_per_iter=steps_per_iter,n_iter=n_iter,n_loops=n_loops,n_tests=n_tests,
          cat=cat,classlabels=classlabels,baremetal_hostname=baremetal_hostname,label_layer=label_layer,estimate_layer=estimate_layer,
            solverstate=solverstate)
