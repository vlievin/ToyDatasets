import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

COLORS = ['#F44336',"#E91E63",'#9C27B0','#673AB7','#3F51B5','#2196F3','#03A9F4','#00BCD4','#4CAF50',
 '#8BC34A','#CDDC39','#FFEB3B','#FFC107','#FF9800','#FF5722']


R_MIN = 12
R_MAX = 36
line_width_min = 2
line_width_max = 4
background_intensity = 30.0 / 255.0

def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

def DrawRandomCircle(img,segments,r_min,r_max,alpha):
    color = hex2rgb( np.random.choice(COLORS) )
    t = np.random.random()
    r = int(t * r_min + (1-t) * r_max)
    ti = np.random.random()
    tj = np.random.random()
    i = int(ti*img.shape[0])
    j = int(tj*img.shape[1])
    canvas = img.copy()
    cv2.circle(canvas,(i,j), r, color, -1)
    cv2.circle(segments,(i,j), r, (1,0,0), -1)
    img = cv2.addWeighted(img, 1.0 - alpha, canvas, alpha, 0, img )
    box  = [i-r,j-r,2*r,2*r]
    return img,segments,box

def DrawRandomSquare(img,segments,r_min,r_max,alpha):
    color = hex2rgb( np.random.choice(COLORS) )
    t = np.random.random()
    r = int(t * r_min + (1-t) * r_max)
    i = int(np.random.random()*img.shape[0])
    j = int(np.random.random()*img.shape[1])
    theta = np.pi * np.random.random()
    ri = r*np.cos(theta)
    rj = r*np.sin(theta) 
    pts = [(ri,rj),(-rj,ri),(-ri,-rj),(rj,-ri) ]
    pts = [(i+y,j+x) for (y,x) in pts]
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    canvas = img.copy()
    cv2.fillPoly(canvas,[pts],color)
    cv2.fillPoly(segments,[pts],(0,1,0))
    img = cv2.addWeighted(img, 1.0 - alpha, canvas, alpha, 0, img )
    box = [min(pts[:,:,0])[0],min(pts[:,:,1])[0], max(pts[:,:,0])[0]-min(pts[:,:,0])[0], max(pts[:,:,1])[0] - min(pts[:,:,1])[0] ]
    return img,segments,box

def DrawRandomLine(img,segments,line_width_min,line_width_max,alpha):
    color = hex2rgb( np.random.choice(COLORS) )
    t = np.random.random()
    line_width = int(t * line_width_min + (1-t) * line_width_max)
    i1 = int(np.random.random()*img.shape[0])
    j1 = int(np.random.random()*img.shape[1])
    i2 = int(np.random.random()*img.shape[0])
    j2 = int(np.random.random()*img.shape[1])
    canvas = img.copy()
    cv2.line(canvas,(i1,j1),(i2,j2),color,line_width)
    cv2.line(segments,(i1,j1),(i2,j2),(0,0,1),line_width)
    img = cv2.addWeighted(img, 1.0 - alpha, canvas, alpha, 0, img )
    pts = np.asarray([(i1,j1),(i2,j2)])
    box = [min(pts[:,0]),min(pts[:,1]), max(pts[:,0])-min(pts[:,0]), max(pts[:,1]) - min(pts[:,1]) ]
    return img,segments,box

def generateSegmentation(canvas_size, n_max, alpha = 0.5, noise_types=[], r_min=R_MIN,r_max=R_MAX):
    background_intensity = np.clip( np.random.normal(80.0 / 255.0, 40.0 / 255.0) , 0,1)
    canvas = background_intensity * np.ones((canvas_size,canvas_size,3))
    segments = np.zeros((canvas_size,canvas_size,3))
    boxes = []
    labels = []
    for _ in range(np.random.choice(range(n_max))):
        canvas,segments,b = DrawRandomCircle(canvas,segments,r_min,r_max,alpha)
        boxes += [b]
        labels += [1]
    for _ in range(np.random.choice(range(n_max))):
        canvas,segments,b = DrawRandomSquare(canvas,segments,r_min,r_max,alpha)
        boxes += [b]
        labels += [2]
    for _ in range(np.random.choice(range(n_max))):
        canvas,segments,b = DrawRandomLine(canvas,segments,line_width_min,line_width_max,alpha)
        boxes += [b]
        labels += [3]
    for t in noise_types:
        canvas = noisy(t,canvas)
    return canvas,segments, labels, boxes

def generateClassification(canvas_size, alpha = 0.5, noise_types=[]):
    canvas = background_intensity * np.ones((canvas_size,canvas_size,3))
    segments = np.zeros((canvas_size,canvas_size,3))
    label = np.random.choice(3)
    if label ==0:
        canvas,_,box = DrawRandomCircle(canvas,segments,r_min,r_max,alpha)
    elif label == 1:
        canvas,_,box = DrawRandomSquare(canvas,segments,r_min,r_max,alpha)
    elif label == 2:
        canvas,_,box = DrawRandomLine(canvas,segments,line_width_min,line_width_max,alpha)
    for t in noise_types:
        canvas = noisy(t,canvas)
    label += 1
    return canvas,label,box

def generateDetection(canvas_size, n_max, alpha = 0.5, noise_types=[]):
    canvas = background_intensity * np.ones((canvas_size,canvas_size,3))
    segments = np.zeros((canvas_size,canvas_size,3))
    boxes = []
    labels = []
    for _ in range(max(1,np.random.choice(range(n_max)))):
        canvas,segments,b = DrawRandomCircle(canvas,segments,r_min,r_max,alpha)
        boxes += [b]
        labels += [1]
    for _ in range(max(1,np.random.choice(range(n_max)))):
        canvas,segments,b = DrawRandomSquare(canvas,segments,r_min,r_max,alpha)
        boxes += [b]
        labels += [2]
    for _ in range(max(1,np.random.choice(range(n_max)))):
        canvas,segments,b = DrawRandomLine(canvas,segments,line_width_min,line_width_max,alpha)
        boxes += [b]
        labels += [3]
    for t in noise_types:
        canvas = noisy(t,canvas)
    return canvas,segments, labels, boxes

def stackSegments(segments):
    canvas = np.zeros((segments.shape[:2]))
    canvas += 1 * segments[:,:,0]
    canvas += 2 * segments[:,:,1]
    canvas += 3 * segments[:,:,2]
    return canvas

class SimpleSegmentationDataset(Dataset):
    """A simple dataset for image segmentation purpose"""
    def __init__(self, patch_size, n_max, alpha =1.0,virtual_size=1000, stack=True, r_min=R_MIN, r_max=R_MAX):
        self.r_min = r_min
        self.r_max = r_max
        self.virtual_size = virtual_size
        self.patch_size = patch_size
        self.n_max = n_max
        self.alpha = alpha
        self.stack = stack

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        x,y,_,_ = generateSegmentation(self.patch_size, self.n_max, self.alpha,r_min=self.r_min,r_max=self.r_max)
        # none leayer for the segmentation
        none_layer = (1 - (y.sum(2) > 0 )).astype(np.uint8)[:,:,np.newaxis]
        y = np.concatenate([none_layer, y], axis = 2)
        # Torch format
        x = x.transpose([2,0,1])
        y = y.transpose([2,0,1])
        if self.stack:
            y = stackSegments(y)
            y = y[np.newaxis,:,:]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x,y
    
class SimpleClassificationDataset(Dataset):
    """
    A simple dataset for image classification purpose"""
    def __init__(self, patch_size, alpha =1.0,virtual_size=1000):
        self.virtual_size = virtual_size
        self.patch_size = patch_size
        self.alpha = alpha
        
    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        x,y,box = generateClassification(self.patch_size, self.alpha)
        x = x.transpose([2,0,1])
        x = torch.from_numpy(x).float()
        #y = torch.from_numpy(y).long()
        return x,y
    
class SimpleDetectionDataset(Dataset):
    """
    Work in Progess
    
    A simple dataset for image classification purpose"""
    def __init__(self, patch_size, n_max, alpha =1.0,virtual_size=1000):
        self.virtual_size = virtual_size
        self.patch_size = patch_size
        self.alpha = alpha
        self.n_max = n_max
        
    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        x,y,box = generateClassification(self.patch_size, self.alpha)
        x = x.transpose([2,0,1])
        x = torch.from_numpy(x).float()
        #y = torch.from_numpy(y).long()
        box = torch.from_numpy(np.asarray(box)).float()
        return x,y,box

def drawBox(img,box, text=None):
    img = img.astype(np.uint8).copy() 
    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255, 235, 59),2)
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img,text,(int(box[0]),int(box[1]-5)), font, 0.5,(255, 235, 59),2,cv2.LINE_AA)
    return img

