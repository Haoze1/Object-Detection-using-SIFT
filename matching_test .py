
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift,estimate_bandwidth
import time
from multiprocessing import Process,Pool
from joblib import Parallel,delayed


# In[2]:


def produce_bounding_box(corners,query_positions,train_positions):
    query_output=[]
    train_output=[]
    for i in range(0,len(query_positions)):
        if ((not query_positions in query_output) and (not train_positions in train_output) ):
            query_output.append(query_positions[i])
            train_output.append(train_positions[i])
    query_points=np.float32(query_output).reshape(-1,1,2)
    train_points=np.float32(train_output).reshape(-1,1,2) 
    M, mask = cv.findHomography(query_points, train_points, cv.RANSAC,5.0)
    if(M is None):
        M, mask = cv.findHomography(query_points, train_points)
    corners=np.float32(corners).reshape(-1,1,2)
    dst=cv.perspectiveTransform(corners,M)
    print(dst)
    return dst


# In[3]:


def matched_positions(img1,img2,k=3,n=2,unique=False,order=0):
    '''
    tips: works better for scale within 0.5-2
    Args:
        img1: template image
        img2: target image
        k: k matches for each feature point
        n: filter for filtering out points with distance > n* min_distance
        unique: whether the object is unique 
        
    return:
        positions: first n best matches positions
    side effects:
        create a result_with_bb.png showing the matched result with bounding box
    '''
    # Initiate SIFT detector
    if unique:
        k=1
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    print("# of keypoints: ",len(kp1))
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k)
    matches_flat=[]
    # flat the match list
    for match in matches:
        for match_point in match:
            matches_flat.append(match_point)
    matches_flat=sorted(matches_flat,key=lambda x: x.distance)
    max_distance=matches_flat[len(matches_flat)-1].distance
    min_distance=matches_flat[0].distance
    good=[]
    for match in matches_flat:
       if match.distance<=max(0.02,n*min_distance):
            good.append(match)
    #extract position in the target image
    if len(good)<5 :
        good=matches_flat[:5]  

    if not unique:
        positions_map = map(lambda x:(x.queryIdx,x.trainIdx,kp2[x.trainIdx].pt) , good)
        positions=map(lambda x:x[2],positions_map)
        bandwidth =max(50,estimate_bandwidth(positions, quantile=0.3,n_jobs=-1))
        if (len(good)<=20):
            bandwidth=min(50,bandwidth)
        img1_width_up=2*max(img1.shape)
        print("upper_width",img1_width_up)
        bandwidth=min(img1_width_up,bandwidth)
        img1_width_low=max(img1.shape)
        bandwidth=max(img1_width_low,bandwidth)
        print("lower_width",img1_width_low)
        print('bandwidth: ',bandwidth)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(positions)
        labels=ms.labels_
        labels_unique=np.unique(labels)
        print("# of clusters: ",len(labels_unique))
        print("labels: ",labels)
        n_clusters=len(labels_unique)
        matched_group = []
        for i in range(0,n_clusters):
            matched_group.append({'query':[],'train':[]})
        for i in range(0,len(positions_map)):
            for j in range(0,n_clusters):
                if labels[i]==j:
                    matched_group[j]['query'].append(kp1[positions_map[i][0]].pt)
                    matched_group[j]['train'].append(kp2[positions_map[i][1]].pt)
        h,w=img1.shape
        corners=[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,[good],None, flags=2)
        for i in range(0,n_clusters):
            if len(matched_group[i]['query'])>=5:
                try:
                    box=produce_bounding_box(corners,matched_group[i]['query'],matched_group[i]['train'])
                    # translate the box to the correct positions
                    box = map(lambda x:[[x[0][0]+w,x[0][1]]],box)
                    img3 = cv.polylines(img3,[np.int32(box)],True,255,3, cv.LINE_AA)
                except:
                    print("Not enough information to produce the bounding box")
        print("result_with_bb{}.png".format(order))
        cv.imwrite("result_with_bb{}.png".format(order),img3)
        return positions
    else:
        # make every matched point has a unique query point,use the matched point with best distance
        output=[]
        present_train_idx=[]
        present_query_idx=[]
        for element in good:
            if (element.trainIdx not in present_train_idx) and (element.queryIdx not in present_query_idx):
                present_train_idx.append(element.trainIdx)
                present_query_idx.append(element.queryIdx)
                output.append(element)
        good=output
        positions_map = map(lambda x:(x.queryIdx,x.trainIdx,kp2[x.trainIdx].pt) , good)
        positions=map(lambda x:x[2],positions_map)
        query_positions=map(lambda x: kp1[x[0]].pt,positions_map)
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,[good],None, flags=2)
        h,w=img1.shape
        print(h,w)
        corners=[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]
        box=produce_bounding_box(corners,query_positions,positions)
        box = map(lambda x:[[x[0][0]+w,x[0][1]]],box)
        img3=cv.polylines(img3,[np.int32(box)],True,255,3, cv.LINE_AA)
        print("result_with_bb{}.png".format(order))
        cv.imwrite("result_with_bb{}.png".format(order),img3)
        return positions        
        



# In[ ]:
num_processes=4
img1=cv.imread(r"images\graf\stair.png",0)
def work(x):
    img2=cv.imread(r"images\graf\RiverBank\p{}.png".format(x+1),0)
    positions=matched_positions(img1,img2,n=2.5,k=3,unique=False,order=x+1)

parallelizer=Parallel(n_jobs=num_processes,backend="threading")
tasks_iterator=(delayed(work)(x) for x in range(0,10))
start=time.time()
result = parallelizer(tasks_iterator)
end=time.time()
print("Running time: ", end-start)
'''if __name__ == '__main__':
    p=Pool(num_processes)
    start=time.time()
    p.map(work,range(0,10))
    end=time.time()
    print("Running time: ", end-start)

start=time.time()
for i in range(0,10):
    work(i)
end=time.time()
print("Running time: ", end-start)
'''