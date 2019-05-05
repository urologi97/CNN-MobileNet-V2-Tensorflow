

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# module level variables ##############################################################################################
TRAINING_IMAGES_DIR = os.getcwd() + "/training_images/"
TEST_IMAGES_DIR = os.getcwd() + "/test_images/"

MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING = 10
MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING = 100

MIN_NUM_IMAGES_REQUIRED_FOR_TESTING = 3

TRAINING_DATA_DIR = os.getcwd() + "/" + "training_data"
TRAIN_CSV_FILE_LOC = TRAINING_DATA_DIR + "/" + "train_labels.csv"
EVAL_CSV_FILE_LOC = TRAINING_DATA_DIR + "/" + "eval_labels.csv"

#######################################################################################################################
def main():
    if not checkIfNecessaryPathsAndFilesExist():
        return
    
    try:
        if not os.path.exists(TRAINING_DATA_DIR):
            os.makedirs(TRAINING_DATA_DIR)

    except Exception as e:
        print("unable to create directory " + TRAINING_DATA_DIR + "error: " + str(e))
    


    # mengubah file-file xml training ke csv
    print("converting xml training data . . .")
    trainCsvResults = xml_to_csv(TRAINING_IMAGES_DIR)
    trainCsvResults.to_csv(TRAIN_CSV_FILE_LOC, index=None)
    print("training xml to .csv conversion successful, saved result to " + TRAIN_CSV_FILE_LOC)

    # mengubah file-file xml testing ke csv
    print("converting xml test data . . .")
    testCsvResults = xml_to_csv(TEST_IMAGES_DIR)
    testCsvResults.to_csv(EVAL_CSV_FILE_LOC, index=None)
    print("test xml to .csv conversion successful, saved result to " + EVAL_CSV_FILE_LOC)


#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TRAINING_IMAGES_DIR):
        print('')
        print('ERROR: the training images directory "' + TRAINING_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the training images?')
        print('')
        return False

    
    trainingImagesWithAMatchingXmlFile = []
    for fileName in os.listdir(TRAINING_IMAGES_DIR):
        if fileName.endswith(".jpg"):
            xmlFileName = os.path.splitext(fileName)[0] + ".xml"
            if os.path.exists(os.path.join(TRAINING_IMAGES_DIR, xmlFileName)):
                trainingImagesWithAMatchingXmlFile.append(fileName)

    
    if len(trainingImagesWithAMatchingXmlFile) <= 0:
        print("ERROR: there don't seem to be any images and matching XML files in " + TRAINING_IMAGES_DIR)
        print("Did you set up the training images?")
        return False

    
    if len(trainingImagesWithAMatchingXmlFile) < MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING:
        print("ERROR: there are not at least " + str(MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING) + " images and matching XML files in " + TRAINING_IMAGES_DIR)
        print("Did you set up the training images?")
        return False

    
    if len(trainingImagesWithAMatchingXmlFile) < MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING:
        print("WARNING: there are not at least " + str(MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING) + " images and matching XML files in " + TRAINING_IMAGES_DIR)
        print("At least " + str(MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING) + " image / xml pairs are recommended for bare minimum acceptable results")

    if not os.path.exists(TEST_IMAGES_DIR):
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        return False

    
    testImagesWithAMatchingXmlFile = []
    for fileName in os.listdir(TEST_IMAGES_DIR):
        if fileName.endswith(".jpg"):
            xmlFileName = os.path.splitext(fileName)[0] + ".xml"
            if os.path.exists(os.path.join(TEST_IMAGES_DIR, xmlFileName)):
                testImagesWithAMatchingXmlFile.append(fileName)

    
    if len(testImagesWithAMatchingXmlFile) <= 3:
        print("ERROR: there are not at least " + str(MIN_NUM_IMAGES_REQUIRED_FOR_TESTING) + " image / xml pairs in " + TEST_IMAGES_DIR)
        print("Did you separate out the test image / xml pairs from the training image / xml pairs?")
        return False

    return True

#######################################################################################################################
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text, int(root.find('size')[0].text), int(root.find('size')[1].text), member[0].text,
                     int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
            xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

#######################################################################################################################
if __name__ == "__main__":
    main()
