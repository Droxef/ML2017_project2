# ML2017_project2
Containing a run.py main file, and sub-modules: helpers.py and extract_features

## External Libraries
correct versions are in the file requirement.txt
Run on Windows 10 and MacOS  High Sierra
## Working space
All python scripts must be on the same level, and a directory dataset must be added, containing a "training" directory and a "test_set_images".
All test images must be individually in folder with the same name as the image.
In the training directory, there must be two folders: groundtruth, and images.
The images size must be a multiple of 16 pixels

The model is saved under "svm_model.pkl"
The pca model is saved under "pca_model.pkl"
computed features are saved under "feature_all_patches.txt"
## Running the code
lauch run.py to get a "submission.csv" for kaggle.
Several flags can be added: 
* trained: will used the trained model and compute only the features of the testing images
* features: will load the computed features and train the model with them
* test_features: was only for debug

a second argument can be given, the number of testing images

## Time estimation
For each part of the code, a progression bar, or a time remaining text will show, thanks to tqdm library and scikit-learn verbose
* Estimation of features extraction: *2h per 50 images*
* Estimation of PCA: *1 minute*
* Estimation of model training: *10 minutes* for 20000 iters
