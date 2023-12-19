# Carbon-steel-classification
Carbon-steel-classification based on DL and UHCS database 

# Pipeline
  1. Data collecting and preprocessing
  2. Decide on and build the Network(ResNet, VGG16 etc.)
  3. Training
  4. Assessing and tuning(OpenVINO)
  5. Deployment on PC
# Dataset 
  Data used in this repo is available on https://www.kaggle.com/datasets/safi842/highcarbon-micrographs/
  
  Further information about the dataset: https://link.springer.com/article/10.1007/s40192-017-0097-0#Sec3
# Preprocessing of the UHCS Database
  The UHCSDB contains 961 high-carbon-steel micrographs with 598 labeled by expert(see in metadata.xlsx), to retain expected form of the dataset, we distribute the original dataset into four categories(Pearlite, Spheroidite, Network and Martensite), and for the rest 363 micrographs, we manually labeled them to form the final dataset of our carbon-steel classification task.

  Considering the repeated features of each micrograph, we use OpenCV to cut each micrograph(645 x 481 pixels) into eight sub-micrographs with the same size(80 x 60 pixels) to enlarge our dataset, which may promote the performance of the netwrok.

  The dataset is diveded into three parts: training data, validation data and test data with a proportion of 8 : 1 : 1.

  
