# Carbon-steel-classification
Carbon-steel-classification based on DL and UHCS database 

# Pipeline
  1. Data collecting and preprocessing
  2. Decide on the Network(ResNet, VGG16 etc.)
  3. Training
  4. Assessing and tuning(OpenVINO)
  5. Deployment on PC
# Dataset 
  Data used in this repo is available on https://www.kaggle.com/datasets/safi842/highcarbon-micrographs/
  
  Further information about the dataset: https://link.springer.com/article/10.1007/s40192-017-0097-0#Sec3
# Preprocessing of the UHCS Database
  The UHCSDB contains 961 high-carbon-steel micrographs with 598 labeled by expert(see in metadata.xlsx), to gain the wanted form of the dataset, we distribute the original dataset into 
