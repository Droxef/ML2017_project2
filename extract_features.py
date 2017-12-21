""" Module used to extract the features of a 16x16 px patch of an image """
import warnings
import numpy as np
import scipy
from scipy import signal
import skimage
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops

WINDOW=16

sobel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
def edgedetect(img):
    """ 
        edge detection of patch. Return gradient of image
        Use of scipy.signal library for its convolution function
    """
    g1 = scipy.signal.convolve2d(img,sobel,'same')
    g2 = scipy.signal.convolve2d(img,sobel.T,'same')
    return np.abs(g1+g2)

distances    = [1]
orientations = [0,np.pi/4,np.pi/2,3*np.pi/4]
num_glcm_features=5 
def texture_feat(img):
    """ 
        Return the texture of a patch using the gray level co-occurence matrices
        http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img=skimage.img_as_ubyte(img)
    glcm=greycomatrix(img,distances,orientations,256,symmetric=True, normed=True)
    diss=greycoprops(glcm,'dissimilarity').flatten()
    contrast=greycoprops(glcm,'contrast').flatten()
    correl = greycoprops(glcm,'correlation').flatten()
    homogen =greycoprops(glcm,'homogeneity').flatten()
    energy = greycoprops(glcm,'energy').flatten()
    del glcm
    feat=np.hstack((contrast,diss,correl,homogen,energy))
    return feat

def extract_features_patch(img):
    """
        Extract features of a single patch given:
        - mean of all channels
        - variance of all channels
        if gray-level:
        - textures
        - edge density
    """
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    ###### graylevel features #########
    if(len(img.shape)<3):
        feat_text = texture_feat(img)
        feat_ed = np.mean(edgedetect(img),axis=(0,1))
        feat=np.hstack((feat,feat_text,feat_ed))
    return feat

def extract_features_ngbr(color,gray,index):
    """
        Extract features of neighboring patches, and add them to patch feature
    """
    if(color.shape[0]%100==0):
        color=color.reshape((100,int(np.sqrt(color.shape[0]//100)),int(np.sqrt(color.shape[0]//100)))+color.shape[1:])
        gray=gray.reshape((100,int(np.sqrt(gray.shape[0]//100)),int(np.sqrt(gray.shape[0]//100)))+gray.shape[1:])
    else:
        color=color.reshape((1,int(np.sqrt(color.shape[0])),int(np.sqrt(color.shape[0])))+color.shape[1:])
        gray=gray.reshape((1,int(np.sqrt(gray.shape[0])),int(np.sqrt(gray.shape[0])))+gray.shape[1:])
    k=index//(color.shape[1]*color.shape[2])
    row,col=np.unravel_index(index%100,(color.shape[1]-1,color.shape[2]-1))
    feat=np.append(extract_features_patch(color[k,row+1,col+1]),extract_features_patch(gray[k,row+1,col+1]))
    for i in [-1,1]:
        feat=np.append(feat,extract_features_patch(color[k,row+1,col+1+i]))
        feat=np.append(feat,extract_features_patch(gray[k,row+1,col+1+i]))
        feat=np.append(feat,extract_features_patch(color[k,row+1+i,col+1]))
        feat=np.append(feat,extract_features_patch(gray[k,row+1+i,col+1]))
    return feat