import os
COMPUTER_NAME = os.environ['COMPUTERNAME']
print("Computer: ", COMPUTER_NAME)

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320

NDSB3_RAW_SRC_DIR = "C:/data/kaggle/dsb17/original/dsb17/stage1/"
LUNA16_RAW_SRC_DIR = "E:/data/kaggle/dsb17/original/luna/"

NDSB3_EXTRACTED_IMAGE_DIR = "E:/data/kaggle/dsb17/processed/dsb17_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = "E:/data/kaggle/dsb17/processed/luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR = "E:/data/kaggle/dsb17/processed/dsb3_nodule_predictions/"

