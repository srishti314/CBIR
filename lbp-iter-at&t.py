# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:30:53 2017

@author: Shristi Gupta
"""

%pylab inline --no-import-all
# OpenCV bindings
import cv2
# To performing path manipulations 
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
# Utility package -- use pip install cvutils to install
from matplotlib import pyplot as plt
import cvutils
# To read class from file
import csv
import pandas as pd

# Store the path of training images in train_image/
train_images = cvutils.imlist("C:/Users/Shristi Gupta/Desktop/research-AI/python codes/images-corel1K/")
# Dictionary containing image paths as keys and corresponding label as value

Y_test = []
Y_name = []

##### testing lbp code on the database

for train_image in train_images:
    # Read the image
    im = cv2.imread(train_image)
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 1
    # Number of points to be considered as neighbourers 
    no_points = 8 * radius
    # Uniform LBP is used
    #lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    
    lbp_value=[[0 for x in range(0,len(im_gray[0]))] for y in range(0,len(im_gray))]
    lbp=[[[0,0,0,0,0,0,0,0] for x in range(0,len(im_gray[0]))] for y in range(0,len(im_gray))]
    
    for i in range(1,len(im_gray)-1):
        for j in range(1,len(im_gray[0])-1):
            pat=[0]*8
            pat[0]=int(im_gray[i][j+1])-int(im_gray[i][j])
            pat[1]=int(im_gray[i-1][j+1])-int(im_gray[i][j])
            pat[2]=int(im_gray[i-1][j])-int(im_gray[i][j])
            pat[3]=int(im_gray[i-1][j-1])-int(im_gray[i][j])
            pat[4]=int(im_gray[i][j-1])-int(im_gray[i][j])
            pat[5]=int(im_gray[i+1][j-1])-int(im_gray[i][j])
            pat[6]=int(im_gray[i+1][j])-int(im_gray[i][j])
            pat[7]=int(im_gray[i+1][j+1])-int(im_gray[i][j])
            
            for k in range(0,8):
                if pat[k]>=0:
                    lbp[i][j][k]=1
                else:
                   lbp[i][j][k]=0 
                    
                lbp_value[i][j]=lbp_value[i][j]+lbp[i][j][k]*(2**k)
            
    lbp_arr = np.array(lbp_value)
    x1 = itemfreq(lbp_arr.ravel())
    x=[[0,0] for m in range(0,256)]
     
    if (len(x1)==256):
        x=x1
    else:
        for i in range(0,256):
            if i in x1[:,0]:
                x[i][0]=i
                [x[i][1]] =[x1[j][1] for j, row in enumerate(x1) if i == int(row[0])]
            else:
                x[i]=[i,0]
                
    x=np.array(x)
    histx = x[:, 1]
        
    #appending the two histograms
    #hist = np.concatenate((histx, histy), axis=0)
    chans = cv2.split(im)
    colors = ("b", "g", "r")
    features = []
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        hist1 = cv2.calcHist([chan], [0], None, [8], [0, 256])
        #normalizing the vector
        hist2 = hist1[:, 0]
        features.extend(hist2)
        
    hist = np.concatenate((histx, np.array(features).ravel()), axis=0)
    Y_name.append(train_image)
    Y_test.append(hist)
    # Append class label in y_test
    #y_test.append(train_dic[os.path.split(train_image)[1]])
    
#comparing the histograms using Chi squared value
# lower the value better will be the match
score_lbp=[[0 for x in range(0,len(Y_name))] for y in range(0,len(Y_name))]

for indexx, x in enumerate(Y_test):
    for indexy, y in enumerate(Y_test):
        score_lbp[indexx][indexy] = sum(abs((x[i]-y[i])/(1+x[i]+y[i])) for i in range(0,280))

#### keeping number of images retrieved constant
n=100
indices_lbp=[[] for x in range(0,len(Y_name))]

for j in range (0,len(Y_name)):
    df0 = pd.DataFrame(score_lbp[j])
    df0.sort_values(by=[0],inplace = True)
    df2=df0.reset_index()
    for i in range (0,n):
        indices_lbp[j].append(df2['index'][i])

tot_lbp=[0 for x in range(0,len(Y_name))]

for i in range (0,len(Y_name)):
    x=i//100
    tot_lbp[i]=sum(1 for j in range(0,len(indices_lbp[i])) if (indices_lbp[i][j]<100+100*x and indices_lbp[i][j]>=100*x) )
         
df3 = pd.DataFrame(tot_lbp)
df3.to_excel('tot_lbp.xlsx', header=False, index=False)