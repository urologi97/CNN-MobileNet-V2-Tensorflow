import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

# module-level variables ##############################################################################################

# memasukkan csv training dan lokasi gambar training
TRAIN_CSV_FILE_LOC = os.getcwd() + "/training_data/" + "train_labels.csv"
TRAIN_IMAGES_DIR = os.getcwd() + "/training_images"

# memasukkan csv testing dan lokasi gambar testing
EVAL_CSV_FILE_LOC = os.getcwd() + "/training_data/" + "eval_labels.csv"
TEST_IMAGES_DIR = os.getcwd() + "/test_images"

# memberikan lokasi tempat tfrecord akan dibuat
TRAIN_TFRECORD_FILE_LOC = os.getcwd() + "/training_data/" + "train.tfrecord"
EVAL_TFRECORD_FILE_LOC = os.getcwd() + "/training_data/" + "eval.tfrecord"

#######################################################################################################################
def main():
    if not checkIfNecessaryPathsAndFilesExist():
        return

    # membuat tfrecord training
    trainTfRecordFileWriteSuccessful = writeTfRecordFile(TRAIN_CSV_FILE_LOC, TRAIN_TFRECORD_FILE_LOC, TRAIN_IMAGES_DIR)
    if trainTfRecordFileWriteSuccessful:
        print("successfully created the training TFRectrds, saved to: " + TRAIN_TFRECORD_FILE_LOC)

    # Membuat tfrecord testing
    evalTfRecordFileWriteSuccessful = writeTfRecordFile(EVAL_CSV_FILE_LOC, EVAL_TFRECORD_FILE_LOC, TEST_IMAGES_DIR)
    if evalTfRecordFileWriteSuccessful:
        print("successfully created the eval TFRecords, saved to: " + EVAL_TFRECORD_FILE_LOC)

#######################################################################################################################
def writeTfRecordFile(csvFileName, tfRecordFileName, imagesDir):
    # membaca file csv
    csvFileDataFrame = pd.read_csv(csvFileName)

    # mengubah format csv
    csvFileDataList = reformatCsvFileData(csvFileDataFrame)

    # menjalankan trRecordWriter untuk membuat tf record
    tfRecordWriter = tf.python_io.TFRecordWriter(tfRecordFileName)

    #mengubah csv menjadi tfrecord
    for singleFileData in csvFileDataList:
        tfExample = createTfExample(singleFileData, imagesDir)
        tfRecordWriter.write(tfExample.SerializeToString())
    
    tfRecordWriter.close()
    return True

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TRAIN_CSV_FILE_LOC):
        print('ERROR: TRAIN_CSV_FILE "' + TRAIN_CSV_FILE_LOC + '" does not seem to exist')
        return False
    

    if not os.path.exists(TRAIN_IMAGES_DIR):
        print('ERROR: TRAIN_IMAGES_DIR "' + TRAIN_IMAGES_DIR + '" does not seem to exist')
        return False
    

    if not os.path.exists(EVAL_CSV_FILE_LOC):
        print('ERROR: TEST_CSV_FILE "' + EVAL_CSV_FILE_LOC + '" does not seem to exist')
        return False
    

    if not os.path.exists(TEST_IMAGES_DIR):
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        return False
    

    return True


#######################################################################################################################
# mengubah csv menjadi tuple list
def reformatCsvFileData(csvFileDataFrame):
    
    dataFormat = namedtuple('data', ['filename', 'object'])

    csvFileDataFrameGroupBy = csvFileDataFrame.groupby('filename')

    csvFileDataList = []
    for filename, x in zip(csvFileDataFrameGroupBy.groups.keys(), csvFileDataFrameGroupBy.groups):
        csvFileDataList.append(dataFormat(filename, csvFileDataFrameGroupBy.get_group(x)))
    return csvFileDataList


#######################################################################################################################
def createTfExample(singleFileData, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(singleFileData.filename)), 'rb') as tensorFlowImageFile:
        tensorFlowImage = tensorFlowImageFile.read()

    bytesIoImage = io.BytesIO(tensorFlowImage)
    pilImage = Image.open(bytesIoImage)
    width, height = pilImage.size


    fileName = singleFileData.filename.encode('utf8')
    imageFormat = b'jpg'


    xMins = []
    xMaxs = []
    yMins = []
    yMaxs = []
    classesAsText = []
    classesAsInts = []


    for index, row in singleFileData.object.iterrows():
        xMins.append(row['xmin'] / width)
        xMaxs.append(row['xmax'] / width)
        yMins.append(row['ymin'] / height)
        yMaxs.append(row['ymax'] / height)
        classesAsText.append(row['class'].encode('utf8'))
        classesAsInts.append(classAsTextToClassAsInt(row['class']))



    tfExample = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(fileName),
        'image/source_id': dataset_util.bytes_feature(fileName),
        'image/encoded': dataset_util.bytes_feature(tensorFlowImage),
        'image/format': dataset_util.bytes_feature(imageFormat),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xMins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xMaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(yMins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(yMaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classesAsText),
        'image/object/class/label': dataset_util.int64_list_feature(classesAsInts)}))

    return tfExample


#######################################################################################################################
def classAsTextToClassAsInt(classAsText):
    # membuat kelas kelas pada klasifikasi
    # ubah sesuai kebutuhan kelas

    if classAsText == 'tangga_turun':
        return 1
    elif classAsText == 'kolong_meja_normal':
        return 2
    else:
        print("error in class_text_to_int(), row_label could not be identified")
        return -1

#######################################################################################################################
if __name__ == '__main__':
    main()