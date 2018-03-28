# coding: utf-8
__author__ = 'jeremy'


from PIL import Image
import numpy
a=numpy.load('out_resized.npy')
b=numpy.resize(a,(256,256,21))
c=numpy.resize(a,(21,256,256))
#riddle - is
#x=numpy.resize(a,(256,256,21))
#y=x[:,:,0]  the same as
#x=numpy.resize(a,(21,256,256))
#y=x[0,:,1]  the same as


for i in range(0,21):
    myslice1=c[i,:,:]
    myslice1_scaled=myslice1*255
    myslice1_int=myslice1_scaled.astype('uint8')
    img = Image.fromarray(myslice1_int)
    img.show()
    imrgb=img.convert('RGB')
    imrgb.save('layer'+str(i)+'.jpg')
    imgray=img.convert()
    imgray.save('layergray'+str(i)+'.jpg')

