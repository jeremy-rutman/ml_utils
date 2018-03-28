''' track multiple objects in video simultanesouly - uses
cv2.Tracker_create('KCF') , that can be swapped out
'''

from __future__ import print_function
import cv2
import sys
import sys
import time
import logging
logging.basicConfig(level=logging.DEBUG)
import json
import os

#import skvideo.io
#from skvideo.io import VideoWriter
#from skvideo.io import FFmpegWriter
#zimport numpy

from trendi.utils import imutils
#annoying bald guy tracker using camshift
#  # import the necessary packages
# import numpy as np
# import argparse
# import cv2
#
# from trendi.defense import defense_client


#!/usr/bin/env python

'''
Camshift tracker
================
This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)
http://www.robinhewitt.com/research/track/camshift.html
Usage:
------
    camshift.py [<video source>]
    To initialize tracking, select the object with mouse
Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2

# local module
# import video
# from video import presets

vid='/home/jeremy/projects/soccer/UNC_Jordan_McCrary_Highlights.mp4'
vid='/home/jeremy/projects/soccer/AB700b.mp4'
vid='/home/jeremy/projects/soccer/AB1700_30.mpg'
camera = cv2.VideoCapture(vid)


class App(object):
    def __init__(self, video_src):
        # self.cam = video.create_capture(video_src, presets['cube'])
        # ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')

#        cv2.setMouseCallback('camshift', self.onmouse)
        self.velocity_multiplier = 0.1
        self.velocity_color = [255,0,0]
        self.max_objects = 20
        self.fps = 30
        self.waitKey=100
        self.valid_selections = None
        self.selection = None
        self.selections = [[] for i in range(self.max_objects)]
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None
        self.hists=[[] for i in range(self.max_objects)]
        self.object_being_edited=None
        self.trackertype = 'KCF'
        self.use_static_boxes_if_no_tracking = False
        self.trackers = {}
        self.annotations = []
        self.annotations_file = os.path.join(os.path.dirname(vid),os.path.basename(vid)+'_annotations.json')
        print('annotations file:'+str(self.annotations_file))
        self.load_annotations()
        #these get filled with dicts like:: {'frame':'2','annotations':[{'object_index':object_index,'bbox_xywh':bbox},
 #                                                           {'object':'argentina7','bbox_xywh':[x,y,w,h]}  }
        #                        )
        self.current_annotations = None #this gets filled with [{'object_index':object_index,'bbox_xywh':bbox},
 #                                                           {'object':'argentina7','bbox_xywh':[x,y,w,h]}
    #alternative approach is list of objects and list of bbs for each object
            # [{'object':'brazil1','bbs':[]},
            #             {'object':'brazil2','bbs':[]},
            #             {'object':'brazil3','bbs':[]},
            #             {'object':'brazil4','bbs':[]},
            #             {'object':'brazil5','bbs':[]},
            #             {'object':'brazil6','bbs':[]},
            #             {'object':'brazil7','bbs':[]},
            #             {'object':'brazil8','bbs':[]},
            #             {'object':'brazil9','bbs':[]},
            #             {'object':'brazil10','bbs':[]},
            #             {'object':'brazil_goalie','bbs':[]},
            #             {'object':'argentina1','bbs':[]},
            #             {'object':'argentina1','bbs':[]},
            #             {'object':'argentina1','bbs':[]},


    def show_hist(self):
        print('showhist')
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def show_hists(self,j):
        print('showhists')
        bin_count = self.hists[j].shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hists[j][i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist'+str(j), img)

    def find_annotations_for_frame(self,n_frame):
        i=0
        for annotation in self.annotations:
            if 'frame' in annotation:
                if annotation['frame']==n_frame:
                    return annotation['annotations']
            else:
                pass
#
            #                 print('no frame# in annotation {}'.format(i))
            i=i+1
        print('couldnt find annotation for frame {}'.format(n_frame))
        return []

    def calc_velocities(self,n_frame):
        prev_annotations = self.find_annotations_for_frame(n_frame-2)
        if prev_annotations is None or prev_annotations == []:
            return
        for obj in self.current_annotations:
            has_match=False
            for prev_obj in prev_annotations:
                # print('checking prev obj {} for cont'.format(old_obj['object_index']))
                prev_index = prev_obj['object_index']
                if obj['object_index'] == prev_index:
                    logging.debug('VELOCITY CALC - object {} matches '.format(prev_index))
                    has_match=True
                    break
            if has_match:
                logging.debug('bboxes current {} prev {}'.format(obj['bbox_xywh'],prev_obj['bbox_xywh']))
                (current_x,current_y) = bbox_center(obj['bbox_xywh'])
                (prev_x,prev_y) = bbox_center(prev_obj['bbox_xywh'])
                dt = 1.0/(self.fps)
                dx = current_x-prev_x
                dy = current_y-prev_y
                vx = dx/dt
                vy = dy/dt
                obj['velocity']=[vx,vy]
                logging.debug('v='+str(obj['velocity']))

        self.insert_annotations(n_frame,self.current_annotations)

    def calc_accelerations(self,n_frame):
        prev_annotations = self.find_annotations_for_frame(n_frame-1)
        for obj in self.current_annotations:
            has_match=False
            for prev_obj in prev_annotations:
                # print('checking prev obj {} for cont'.format(old_obj['object_index']))
                prev_index = prev_obj['object_index']
                if obj['object_index'] == prev_index:
                    print('object {} matches ')
                    has_match=True
                    break
            if has_match:
                (current_x,current_y) = bbox_center(obj['bbox_xywh'])
                (prev_x,prev_y) = bbox_center(obj['bbox_xywh'])
                dt = 1.0/(self.fps)
                dx = current_x-prev_x
                dy = current_y-prev_y
                vx = dx/dt
                vy = dy/dt
                obj['v']=[vx,vy]


    def run(self):
        n_frame=0
        movie = 1
        if movie:
            output_moviename='/home/jeremy/projects/soccer/output.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_moviename,fourcc, self.fps, (1280,720))
#        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
            h,w=720,1280

#iusing scikit
#            http://www.scikit-video.org/stable/io.html
#            writer = skvideo.io.FFmpegWriter("outputvideo.mp4", (5, 480, 640, 3))
#            writer = skvideo.io.FFmpegWriter(outputfile, outputdict={'-vcodec': 'libx264', '-b': '300000000','-framerate':'30'})
  #          writer = FFmpegWriter(output_moviename, outputdict={'-framerate':'30'})
            # writer = VideoWriter(output_moviename, frameSize=(1280, 720))
            # writer.open()

        vis_rois=[[] for i in range(self.max_objects)]
        while True:
            (grabbed, frame) = camera.read()

#            ret, self.frame = self.cam.read()
            if not grabbed:
                print('trouble grabbing frame {}'.format(n_frame))
                if movie:
                    out.release()
                break
            self.frame  = frame
            self.n_frame = n_frame
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            vis = self.frame.copy()
            self.orig = self.frame.copy()
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            self.current_annotations = self.find_annotations_for_frame(n_frame)
            print('frame {}, {} annotations'.format(n_frame,len(self.current_annotations)))
            self.print_current_annotations()

         #   print('frame {} annotations {}'.format(n_frame,self.current_annotations))
            prev_annotation = self.find_annotations_for_frame(n_frame-1)
            if prev_annotation == []:
                print('frame {} has no previous annotation'.format(n_frame))
            else:
                for old_obj in prev_annotation:
                   # print('checking prev obj {} for cont'.format(old_obj['object_index']))
                    object_index = old_obj['object_index']
                    old_has_match=False
                    for new_obj in self.current_annotations:
                        if new_obj['object_index'] == object_index:
                       #     print('old obj already has box, deferring')
                            if 'manually_adjusted' in old_obj:
                                print('prev was manually adjusted therefore take it')
                            else:
                                old_has_match=True
                                break

                    if not old_has_match:
             #           print('checking prev annotation {}'.format(old_obj))
                        if object_index in self.trackers:
                            tracker = self.trackers[object_index]
                            print('found tracker for object {}'.format(object_index))
                            ok, bbox = tracker.update(frame)
                            if not ok: #prob with tracker, use previous
                                logging.warning('some problem with tracker for object {}'.format(object_index))
                                bbox = old_obj['bbox_xywh']
                                new_obj['bbox_xywh']=bbox
                                frame = bb_with_text(frame,bbox,'bcop'+str(object_index))
                            else:
                                bbox = [int(bbox[j]) for j in range(len(bbox))]
                               # new_obj['bbox_xywh']=bbox
                                frame = bb_with_text(frame,bbox,'inf '+str(object_index))
                                print('new tracker bbox {}'.format(bbox))

                        else: #no trackers, use previous
                            if self.use_static_boxes_if_no_tracking:
                                bbox = old_obj['bbox_xywh']
                                bbox=[int(bbox[j]) for j in range(len(bbox))]
                                frame = bb_with_text(frame,bbox,'cop '+str(object_index))
                            else:
                                continue
                        self.update_object(self.current_annotations,object_index,bbox)
                 #       print('after update current annotations:{}'.format(self.current_annotations))
                        self.insert_annotations(self.n_frame,self.current_annotations)

            self.calc_velocities(self.n_frame)
            frame = self.draw_bbs(self.current_annotations,image=self.orig.copy())
            cv2.imshow('camshift', frame)
            if movie:
    #            writer.write(frame)
                out.write(frame)

            ch = cv2.waitKey(self.waitKey)
            #print('ch:'+str(ch))
            if ch == 27:
                break
            if ch == ord('p'):
                self.show_backproj = not self.show_backproj

            if ch > ord('0') and ch<ord('9'):
                self.object_being_edited = ch - ord('0')
                if self.selections[self.object_being_edited] == []:
                    time_start=time.time()
                    while(time.time()-time_start<2):
                        time.sleep(0.1)
                    pass #somehow get mouse event here
                self.adjust_box(ch)

            if ch == ord('s'): #Select bbs
                self.select_bbs()

            if ch == ord('e'): #edit bbs
                self.adjust_box()

            if ch == ord('b') or ch == 83 : #go back frame, b or back arrrow
                n_frame = n_frame-1
            if self.annotations == []:
                pass
            else:
                self.insert_annotations(self.n_frame,self.current_annotations)
    #        print('all annotations:{}'.format(self.annotations))
            n_frame+=1
            with open(self.annotations_file,'w') as fp:
                json.dump(self.annotations,fp,indent=4)
                fp.close()
        cv2.destroyAllWindows()

        if movie:
            out.release()
#            writer.release()

    def select_bbs(self):
        i=0
        v=''
        print('starting edit for frame {}, current annotations are {}'.format(self.n_frame,self.current_annotations))
        while v!=ord('q'):

            n=raw_input('object number or (q)uit?')
            if n=='q':
                break

            try:
                object_index = int(n)
            except:
                print('didnt get obj index,using {} '.format(i))
                object_index = i
#                    bbox = cv2.selectROI('camshift', frame, fromCenter)
            bbox = cv2.selectROI('camshift',self.frame, fromCenter=False)
            bbox = [int(bbox[k]) for k in range(len(bbox))]
        # bbox = cv2.selectROI(frame,False)
        #     self.selections[object_index]=[int(bbox[k]) for k in range(len(bbox))]
        #     print('bbox for object {} from selectroi is:{}'.format(object_index,self.selections[i]))
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(self.frame, p1, p2, (0,0,255))

            self.valid_selections = True
            utilized_bb = False

            if self.current_annotations is not []:
                for obj in self.current_annotations:
                    print('checking obj {}'.format(obj))
                    if obj['object_index'] == object_index: #ob ject is already here, replace it
                        obj['bbox_xywh'] = bbox
                        i=i+1
                        continue


            print('new bb')

            tracker = cv2.Tracker_create('KCF') #self.trackertype
            bbox_tuple =  (float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3]))
            logging.debug('frame {} bboxtuple {}'.format(self.orig.shape,bbox_tuple))
            ok = tracker.init(self.orig,bbox_tuple)
            logging.debug('db0')
            if not ok:
                print('problem with tracker init for object {}'.format(object_index))
            else:
                print('tracker for object {} initialized'.format(object_index))
            self.current_annotations.append({'object_index':object_index,'bbox_xywh':bbox})
            self.trackers[object_index] = tracker

            i=i+1
        self.insert_annotations(self.n_frame,self.current_annotations)
        print('frame {} annotations {}'.format(self.n_frame,self.current_annotations))
        print('all annotations {}'.format(self.annotations))

    def load_annotations(self):
        if not os.path.exists(self.annotations_file):
            self.annotations = []
            return
        with open(self.annotations_file,'r') as fp:
            annotations = json.load(fp)
            self.annotations = annotations
            print('found {} annotations, first is {}'.format(len(annotations),annotations[0]))
            fp.close()

    def insert_annotations(self,n_frame,annotations):
        for annotation in self.annotations:
            if annotation['frame']==n_frame:
                annotation['annotations']==annotations
                return
        #frame not found so append it
        dict = {'frame':n_frame,'annotations':annotations}
        self.annotations.append(dict)

    def update_object(self,current_annotations,object_index,bbox,manual=False):
        i=0
        for obj in current_annotations:
            if obj['object_index']==object_index:
                if bbox is None: #no bbox means object is no longer in view
                    print('deleting object {}'.format(object_index))
                    del(current_annotations[i])
                    self.current_annotations = current_annotations
                    return
                else:
                    obj['bbox_xywh']=bbox
                    if manual:
                        obj['manually_adjusted']=True
                self.current_annotations = current_annotations
                return
            i=i+1
    #    print('did not find object {} in current annotations {} '.format(object_index,current_annotations))
        if manual:
            current_annotations.append({'object_index':object_index,'bbox_xywh':bbox,'manually_adjusted':True})
        else:
            current_annotations.append({'object_index':object_index,'bbox_xywh':bbox})
        self.current_annotations = current_annotations
        return

    def adjust_box(self,object_ind=None):
        annotations = self.find_annotations_for_frame(self.n_frame)
        print('editing annotations for {}:{}'.format(self.n_frame,annotations))
        i = 0
        bb = None
        while(1):
            if object_ind is None:
                inp = raw_input('object# to edit, or (q)uit:')
                if inp == 'q':
                    break
                try:
                    object_index = int(inp)
                except:
                    print('could not get number from input, using {}'.format(i))
                    object_index = i
            else:
                object_index = object_ind
            got_object = False
            for annotation in annotations:
                if annotation['object_index'] == object_index:
                    the_annotation = annotation
                    got_object=True
                    logging.debug('got annotation for {}:{}'.format(object_index,the_annotation))
                    break
            if not got_object:
                print('could not find object {} in annotations {}'.format(object_index,annotations))
                continue
            if bb is None:
                bb=the_annotation['bbox_xywh']
            if not bb:
                print('no bb found??')
                continue
            while(1):
                bb_with_text(self.frame,bb,'EDIT '+str(object_index))
                cv2.imshow('camshift',self.frame)
                self.frame=self.orig
                print('(k)ill, arrows or awsd to move, (r)edraw, (q)uit (y) purge 5 frames(z) purge all 5 frames')
                inp = cv2.waitKey(0)
                bb = the_annotation['bbox_xywh']
                h,w = self.frame.shape[0:2]
#                print('inp {} original bb {} h {} w {}'.format(inp,bb,h,w))

                if inp==81 or inp ==ord('a'): #left arrow
                    logging.debug('left')
                    bb[0]=max(0,bb[0]-2)
                elif inp==83 :#ight arrow
                    logging.debug('down')
                    bb[0]=min(w,bb[0]+2)
                elif inp==82 : #up arrow
                    logging.debug('up')
                    bb[1]=max(0,bb[1]-2)
                elif inp==84 : #down arrow
                    logging.debug('down')
                    bb[1]=min(h,bb[1]+2)

                elif inp==ord('d'): #right arrow
                    logging.debug('down')
                    bb[0]=min(w,bb[0]+1)
                elif inp ==ord('w'): #up arrow
                    logging.debug('up')
                    bb[1]=max(0,bb[1]-1)
                elif inp == ord('s'): #down arrow
                    logging.debug('down')
                    bb[1]=min(h,bb[1]+1)
                elif inp==ord('a') : #left arrow
                    logging.debug('left')
                    bb[0]=max(0,bb[0]-1)

                elif inp==ord('r'):
                    bb = cv2.selectROI('camshift',self.frame, fromCenter=False)
                    bb=[int(bb[j]) for j in range(len(bb))]
                    print('new bb from select roi:'+str(bb))
                    break

                elif inp==ord('k'):
                    bb=None
                    break


                elif inp==ord('x'):
                    self.purge_object_from_future(object_index,self.n_frame)
                    return

                elif inp==ord('y'):
                    self.purge_object_from_future(object_index,self.n_frame,how_many=5)
                    return

                elif inp==ord('z'):
                    self.purge_all_from_future(self.n_frame,how_many=5)
                    return


                elif inp==ord('q'):
                    break
 #           print('adjusted bb {} for {}'.format(bb,object_index))
            self.update_object(self.current_annotations,object_index,bb,manual=True)
            self.draw_bbs(self.current_annotations,self.orig.copy())

            self.insert_annotations(self.n_frame,self.current_annotations)

            if bb is not None: #we adjusted bb , didnt kill it, so track it now0
                tracker = cv2.Tracker_create('KCF') #self.trackertype
                bbox_tuple =  (float(bb[0]),float(bb[1]),float(bb[2]),float(bb[3]))
                print('frame {} bboxtuple {}'.format(self.orig.shape,bbox_tuple))
                ok = tracker.init(self.orig,bbox_tuple)
                if not ok:
                    print('problem with tracker init for object {}'.format(object_index))
                else:
                    print('tracker for object {} initialized'.format(object_index))
                self.trackers[object_index] = tracker

    def draw_bbs(self,annotations,image=None):
        for object in annotations:
            bbox = object['bbox_xywh']
            id = object['object_index']
            if image is not None:
                im = image
            else:
                im = self.frame

            self.frame = bb_with_text(im,bbox,str(id))
            if 'velocity' in object:
                (v_x,v_y) = object['velocity']
                (arrow_start_x,arrow_start_y) = bbox_center(bbox)
                (arrow_tip_x,arrow_tip_y) = (int(arrow_start_x+v_x*self.velocity_multiplier),int(arrow_start_y+v_y*self.velocity_multiplier))
                cv2.arrowedLine(im,(arrow_start_x,arrow_start_y),(arrow_tip_x,arrow_tip_y),self.velocity_color,thickness=2)

        return self.frame

    def print_current_annotations(self):
        for annotation in self.current_annotations:
            print(annotation)

    def purge_object_from_future(self,object_index,n_frame,how_many=None):
        for annotation in self.annotations:
            if annotation['frame']<n_frame:
                continue
            if how_many and annotation['frame']-n_frame>how_many:
                continue
#            print('trying to purge from annotation {}'.format(annotation['frame']))
#            print(annotation)
            for i in range(len(annotation['annotations'])):
                obj = annotation['annotations'][i]
                if obj['object_index']==object_index:
                    print('purging {} (lindex {}) from frame {}'.format(object_index,i,annotation['frame']))
                    del(annotation['annotations'][i])
                    annotation_frame = annotation['frame']

                    self.insert_annotations(annotation_frame,annotation)
                    break

    def purge_all_from_future(self,n_frame,how_many=None):
        for annotation in self.annotations:
            if annotation['frame']<n_frame:
                continue
            if how_many and annotation['frame']-n_frame>how_many:
                continue
#            print('trying to purge from annotation {}'.format(annotation['frame']))
#            print(annotation)

            annotation['annotations']=[]
            annotation_frame = annotation['frame']
            self.insert_annotations(annotation_frame,annotation)

def bb_with_text(img_arr,bb_xywh,text,boxcolor=None):
    if boxcolor is None:
        boxcolor = [255,255,0]
    text_color=[200,0,255]
    text_bgnd_color=[255,255,100]
    margin = 20
    cv2.rectangle(img_arr,(bb_xywh[0],bb_xywh[1]),(bb_xywh[0]+bb_xywh[2],bb_xywh[1]+bb_xywh[3]),color=boxcolor,thickness=1)
#text at top
#    img_arr[bb_xywh[1]:bb_xywh[1]+20,bb_xywh[0]:bb_xywh[0]+bb_xywh[2]]=(img_arr[bb_xywh[1]:bb_xywh[1]+20,bb_xywh[0]:bb_xywh[0]+bb_xywh[2]]/2)+np.array(text_bgnd_color)/2
#    cv2.putText(img_arr,text,(bb_xywh[0]+5,bb_xywh[1]+20),cv2.FONT_HERSHEY_PLAIN, 1, text_color)
#text at bottom

    img_arr[bb_xywh[1]+bb_xywh[3]-margin:bb_xywh[1]+bb_xywh[3],bb_xywh[0]:bb_xywh[0]+bb_xywh[2]]=(img_arr[(bb_xywh[1]+bb_xywh[3]-margin):(bb_xywh[1]+bb_xywh[3]),bb_xywh[0]:(bb_xywh[0]+bb_xywh[2])]/2)+np.array(text_bgnd_color)/2
    cv2.putText(img_arr,text,(bb_xywh[0]+2,bb_xywh[1]+bb_xywh[3]-margin/3),cv2.FONT_HERSHEY_PLAIN, 1, text_color)

    return img_arr

def bbox_center(bb_xywh):
    center_x = bb_xywh[0]+bb_xywh[2]/2
    center_y = bb_xywh[1]+bb_xywh[3]/2
    return(center_x,center_y)

def try_opencv_trackers():

    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN

    tracker = cv2.Tracker_create("KCF")
#    cv2.Tra

    # Read video
    vid='/home/jeremy/projects/soccer/UNC_Jordan_McCrary_Highlights.mp4'
    vid='/home/jeremy/projects/soccer/AB2014_11.mp4'
    video = cv2.VideoCapture(vid)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    #bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    print('frame {} bbox {}'.format(frame.shape,bbox))
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255))

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break


if __name__ == '__main__':

    test_opencv_trackers=0

    if test_opencv_trackers:
        try_opencv_trackers()
    else:
        try:
            video_src = sys.argv[1]
        except:
            video_src = 0
        print(__doc__)
        a=App(video_src)
        a.run()




#
# # initialize the current frame of the video, along with the list of
# # ROI points along with whether or not this is input mode
# frame = None
# roiPts = []
# inputMode = False
#
#
# def selectROI(event, x, y, flags, param):
#     # grab the reference to the current frame, list of ROI
#     # points and whether or not it is ROI selection mode
#     global frame, roiPts, inputMode
#
#     # if we are in ROI selection mode, the mouse was clicked,
#     # and we do not already have four points, then update the
#     # list of ROI points with the (x, y) location of the click
#     # and draw the circle
#     if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
#         roiPts.append((x, y))
#         cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
#         cv2.imshow("frame", frame)
#
#
# def main(vid='/home/jeremy/projects/soccer/UNC_Jordan_McCrary_Highlights.mp4'):
#     global frame, roiPts, inputMode
#
#     # construct the argument parse and parse the arguments
#     # ap = argparse.ArgumentParser()
#     # ap.add_argument("-v", "--video",
#     #     help = "path to the (optional) video file")
#     # args = vars(ap.parse_args())
#     #
#     # # grab the reference to the current frame, list of ROI
#     # # points and whether or not it is ROI selection mode
#     #
#     # # if the video path was not supplied, grab the reference to the
#     # # camera
#     # if not args.get("video", False):
#     #     camera = cv2.VideoCapture(0)
#     #
#     # # otherwise, load the video
#     # else:
#     #     camera = cv2.VideoCapture(args["video"])
#
#     camera = cv2.VideoCapture(vid)
#
#     # setup the mouse callback
#     cv2.namedWindow("frame")
#     cv2.setMouseCallback("frame", selectROI)
#
#     # initialize the termination criteria for cam shift, indicating
#     # a maximum of ten iterations or movement by a least one pixel
#     # along with the bounding box of the ROI
#     termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#     roiBox = None
# # keep looping over the frames
#     n_frame=0
#     while True:
#         n_frame=n_frame+1
#         # grab the current frame
#         print('frame {} '.format(n_frame))
#         (grabbed, frame) = camera.read()
#         # check to see if we have reached the end of the
#         # video
#         if not grabbed:
#             break
#
#         # if the see if the ROI has been computed
#         if roiBox is not None:
#             # convert the current frame to the HSV color space
#             # and perform mean shift
#             print('detecting ROI')
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
#
#             # apply cam shift to the back projection, convert the
#             # points to a bounding box, and then draw them
#             (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
#
#             pts = np.int0(cv2.boxPoints(r))
#             cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
#
#                 # show the frame and record if the user presses a key
#             print('roiBox {} roipts {} pts {}'.format(roiBox,roiPts,pts))
#
#         cv2.imshow("frame", frame)
#         key = cv2.waitKey(0) & 0xFF
#
#         # handle if the 'i' key is pressed, then go into ROI
#         # selection mode
#         if key == ord("i") :# and len(roiPts) < 4:
#             # indicate that we are in input mode and clone the
#             # frame
#             inputMode = True
#             orig = frame.copy()
#
#             # keep looping until 4 reference ROI points have
#             # been selected; press any key to exit ROI selction
#             # mode once 4 points have been selected
#             while len(roiPts) < 4:
#                 cv2.imshow("frame", frame)
#                 cv2.waitKey(0)
#
#             # determine the top-left and bottom-right points
#             roiPts = np.array(roiPts)
#             s = roiPts.sum(axis = 1)
#             tl = roiPts[np.argmin(s)]
#             br = roiPts[np.argmax(s)]
#
#             # grab the ROI for the bounding box and convert it
#             # to the HSV color space
#             roi = orig[tl[1]:br[1], tl[0]:br[0]]
#             roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#             #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
#
#             # compute a HSV histogram for the ROI and store the
#             # bounding box
#             roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
#             roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
#             roiBox = (tl[0], tl[1], br[0], br[1])
#
#         # if the 'q' key is pressed, stop the loop
#         elif key == ord("q"):
#             break
#
#         elif key == ord("g"):
#             print('attempting to get answer from yolo')
#             res = defense_client.detect_hls(frame)
#             data = res['data']
#             print('from yolo got {}'.format(data))
#             if len(data)==0:
#                 print('no data in frame ')
#                 continue
#             data = sorted(data,key=lambda object:(-object['confidence']))
#             print('sorted data:'+str(data))
#
#             firstbox = data[0]['bbox']
#             tl = (firstbox[0],firstbox[1])
#             br = (firstbox[0]+firstbox[2],firstbox[1]+firstbox[3])
#
#
#             # grab the ROI for the bounding box and convert it
#             # to the HSV color space
#             orig = frame.copy()
#             roi = orig[tl[1]:br[1], tl[0]:br[0]]
#             roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#             #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
#
#             # compute a HSV histogram for the ROI and store the
#             # bounding box
#             roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
#             roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
#             roiBox = (firstbox[0],firstbox[1],firstbox[0]+firstbox[2],firstbox[1]+firstbox[3])
#
#
#
#         # cleanup the camera and close any open windows
#     camera.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()


