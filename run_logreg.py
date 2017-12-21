import numpy as np
import matplotlib.pyplot as plt
import os,sys
import itertools
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import linear_model
from sklearn import decomposition
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from helpers import *
from extract_features import *
from sklearn.externals import joblib

def pca_decomposition(X_train,X_test):
    """
        PCA decomposition performed on X_train data.
        Once model found with least significant features, transform X_test data as well.
    """
    X_train=standardize(X_train)
    X_test=np.asarray([standardize(x) for x in X_test])
    pca=decomposition.PCA()
    pca.fit(X_train)
    id=np.argmax(np.cumsum(pca.explained_variance_ratio_)[np.cumsum(pca.explained_variance_ratio_)<0.90]) # find number of components
    pca=decomposition.PCA(n_components=id)
    X_pca=pca.fit_transform(X_train)
    X_pca=standardize(X_pca)
    X_test_pca=np.asarray([standardize(pca.transform(x)) for x in X_test])
    return X_pca, X_test_pca

if __name__=="__main__":
    ####### Decision boolean #######
    trained=False
    logistic=False
    features=False

    ####### Use of pretrained model ########
    if(len(sys.argv)>1):
        if(sys.argv[1]=="trained"):
            trained=True
        if(sys.argv[1]=="logistic"):
            logistic=True
        if(sys.argv[1]=="features"):
            features=True
            
    ####### Load train and gt images ########
    print("Loading images...")
    # Loaded a set of images
    root_dir = "dataset/training/"

    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)
    n = min(100, len(files))
    imgs = [load_image(image_dir + files[i]) for i in tqdm(range(n))]

    gt_dir = root_dir + "groundtruth/"
    gt_imgs = [load_image(gt_dir + files[i]) for i in tqdm(range(n))]

    ###### Load test images #######
    test_dir = "dataset/test_set_images/"
    dirs = os.listdir(test_dir)
    if(len(sys.argv)>2):
        n_test=min(int(sys.argv[2]), len(dirs))
    else:
        n_test=min(50, len(dirs))
    
    imgs_test = [load_image(test_dir+dirs[i]+"/"+dirs[i]+".png") for i in tqdm(range(n_test))]
    
    if(not features):
        ###### Get patches for all images ######
        print("Patching the images")
        patches_img=np.asarray([img_patches(imgs[i]) for i in range(len(imgs))])
        patches_img=patches_img.reshape((len(imgs)*patches_img.shape[1]**2,)+patches_img[0,0,0,0].shape)
        patches_glcm=np.asarray([img_patches(imgs[i],gray=True) for i in range(len(imgs))])
        patches_glcm=patches_glcm.reshape((len(imgs)*patches_glcm.shape[1]**2,)+patches_glcm[0,0,0].shape)
        patches_gt=np.asarray([img_patches(gt_imgs[i]) for i in range(len(imgs))])
        patches_gt=patches_gt.reshape((len(imgs)*patches_gt.shape[1]**2,)+patches_gt[0,0,0].shape)

    patches_img_test=np.asarray([img_patches(imgs_test[i]) for i in range(len(imgs_test))])
    patches_img_test=patches_img_test.reshape((
        len(imgs_test),patches_img_test.shape[1]*patches_img_test.shape[2])+patches_img_test[0,0,0,0].shape)
    patches_glcm_test=np.asarray([img_patches(imgs_test[i],gray=True) for i in range(len(imgs_test))])
    patches_glcm_test=patches_glcm_test.reshape((
        len(imgs_test),patches_glcm_test.shape[1]*patches_glcm_test.shape[2])+patches_glcm_test[0,0,0].shape)
    if not features:
        ##### Get feature vector and label vector #####
        print("Finding training feature")
        X=np.asarray([extract_features_ngbr(patches_img,patches_glcm,i) for i in tqdm(range(len(imgs)*(imgs[0].shape[0]//WINDOW)**2))])
        np.savetxt("feature_all_patches_second.txt",X,fmt='%.10e')  # Save all features in file
    print("Finding Testing feature")
    X_test=np.asarray([extract_features_ngbr(patches_img_test[i],patches_glcm_test[i],j) 
                       for i in tqdm(range(patches_img_test.shape[0])) for j in tqdm(range((imgs_test[i].shape[0]//WINDOW)**2))])
    #X_test=np.asarray([extract_features_ngbr(patches_img_test,patches_glcm_test,i+10+2*imgs[0].shape[0]//WINDOW) for i in tqdm(range(len(imgs)*(imgs_test[0].shape[0]//WINDOW)**2))])
    Y=np.asarray([extract_label(patches_gt[i+2]) for i in tqdm(range(len(gt_imgs)*(imgs[0].shape[0]//WINDOW)**2))])
    if(trained):
        # if use pretrained info, get already acquired features
        X=pd.read_csv("feature_all_patches.txt", delimiter=' ', header=None, dtype=np.float).as_matrix()
    
    ##### PCA #####
    print("Running PCA")
    X_pca,X_test_pca=pca_decomposition(X,X_test)
    
    ##### logreg ####
    if(logistic):
        print("splitting data in training and testing set")
        X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.25, random_state=42)
        print("Carrying grid search on hyperparameter, using cross-validation")
        param_grid = {'C': [1e3, 1e4, 1e5, 1e6],}
        clf = GridSearchCV(linear_model.LogisticRegression(C=c, class_weight="balanced"), param_grid, cv=4,verbose=100,iid=False,n_jobs=8)
        clf = clf.fit(X_train,Y_train)
        Y_pred = clf.predict(X_test)
        print(classification_report(Y_test, Y_pred, labels=range(2)))
        with open('logreg_model.pkl','wb') as saved_model:
            print("saving model")
            joblib.dump(clf,saved_model)
    if(trained):
        with open('logreg_model.pkl','rb') as saved_model:
            print("retrieving model")
            clf=joblib.load(saved_model)
        
    ##### Estimating result on test set #####
    Z = [clf.predict(x) for x in X_test_pca]
    masks=[label_to_img(img.shape[0],img.shape[1],Z[i]) for i,img in enumerate(imgs_test)]
    mask_to_submission("submission.csv",masks)