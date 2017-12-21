import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage.color import rgb2gray

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

WINDOW=16
def img_patches(img,WINDOW=WINDOW,gray=False):
    """
        Return a list of patches, with a symmetric padding of size 2*WINDOW in x and y direction.
        If gray is true, will transform a color image in grayscale and return its list of patches.
    """
    REPEAT=1
    if(gray):
        img=rgb2gray(img)
    if(len(img.shape)>2):
        padded_img=skimage.util.pad(img,((REPEAT*WINDOW,REPEAT*WINDOW),(REPEAT*WINDOW,REPEAT*WINDOW),(0,0)),'symmetric')
        patches=skimage.util.view_as_blocks(padded_img,(WINDOW,WINDOW,3))
    else:
        padded_img=skimage.util.pad(img,((REPEAT*WINDOW,REPEAT*WINDOW),(REPEAT*WINDOW,REPEAT*WINDOW)),'symmetric')
        patches=skimage.util.view_as_blocks(padded_img,(WINDOW,WINDOW))
    return patches

def extract_label(img,threshold=0.25):
    value=np.mean(img)
    if value>threshold:
        return 1
    else:
        return 0
    
def standardize(X):
    """ Standardize the data matrix given. shape (num_samples, num_features) """
    X=(X-X.mean(axis=0))/X.std(axis=0)
    return X

def compute_score(Y,Z):
    """ return the F1-score on the class 1 given the correct labels Y, and the predictions Z """
    Zn = np.nonzero(Z)[0]
    Yn = np.nonzero(Y)[0]
    Zp = np.asarray(list(set(range(Z.shape[0]))-set(np.nonzero(Z)[0])))
    Yp = np.asarray(list(set(range(Y.shape[0]))-set(np.nonzero(Y)[0])))

    TP = len(list(set(Yn) & set(Zn)))
    FP = len(list(set(Yp) & set(Zn)))
    FN = len(list(set(Yn) & set(Zp)))
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    score = 2*Precision*Recall/(Precision+Recall)
    print('Score is: {}'.format(score))
    return score

def label_to_img(imgwidth, imgheight, labels, w=WINDOW, h=WINDOW):
    im = np.zeros([imgwidth, imgheight])
    labels=labels.reshape((int(np.sqrt(labels.shape[0])),int(np.sqrt(labels.shape[0])))+labels.shape[1:])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[i+1,j+1]
            idx = idx + 1
    return im

def mask_to_submission(filename,masks):
    """Converts images into a submission file"""
    with open(filename, 'w') as f:
        f.write('id,prediction\n')
        for id_,fn in enumerate(masks):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, id_))
            
def mask_to_submission_strings(image, number):
    """Outputs the strings that should go into the submission file from the prediction mask"""
    step=WINDOW
    for j in range(0, image.shape[1], step):
        for i in range(0, image.shape[0], step):
            label = image[i, j]
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))