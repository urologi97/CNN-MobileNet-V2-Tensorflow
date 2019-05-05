import os
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

# module-level variables ##############################################################################################

INPUT_TYPE = "image_tensor"

INPUT_SHAPE = None

PIPELINE_CONFIG_LOC =  os.getcwd() + "/" + "ssd_mobilenet_v2_coco.config"

TRAINED_CHECKPOINT_PREFIX_LOC = os.getcwd() + "/training_data/model.ckpt-60000"

OUTPUT_DIR = os.getcwd() + "/" + "inference_graph"

#######################################################################################################################
def main(_):
    print("starting script . . .")

    if not checkIfNecessaryPathsAndFilesExist():
        return


    print("calling TrainEvalPipelineConfig() . . .")
    trainEvalPipelineConfig = pipeline_pb2.TrainEvalPipelineConfig()

    print("checking and merging " + os.path.basename(PIPELINE_CONFIG_LOC) + " into trainEvalPipelineConfig . . .")
    with tf.gfile.GFile(PIPELINE_CONFIG_LOC, 'r') as f:
        text_format.Merge(f.read(), trainEvalPipelineConfig)


    print("calculating input shape . . .")
    if INPUT_SHAPE:
        input_shape = [ int(dim) if dim != '-1' else None for dim in INPUT_SHAPE.split(',') ]
    else:
        input_shape = None


    print("calling export_inference_graph() . . .")
    exporter.export_inference_graph(INPUT_TYPE, trainEvalPipelineConfig, TRAINED_CHECKPOINT_PREFIX_LOC, OUTPUT_DIR, input_shape)

    print("done !!")


#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(PIPELINE_CONFIG_LOC):
        print('ERROR: PIPELINE_CONFIG_LOC "' + PIPELINE_CONFIG_LOC + '" does not seem to exist')
        return False

    trainedCkptPrefixPath, filePrefix = os.path.split(TRAINED_CHECKPOINT_PREFIX_LOC)

    if not os.path.exists(trainedCkptPrefixPath):
        print('ERROR: directory "' + trainedCkptPrefixPath + '" does not seem to exist')
        print('was the training completed successfully?')
        return False


    numFilesThatStartWithPrefix = 0
    for fileName in os.listdir(trainedCkptPrefixPath):
        if fileName.startswith(filePrefix):
            numFilesThatStartWithPrefix += 1

    if numFilesThatStartWithPrefix < 3:
        print('ERROR: 3 files statring with "' + filePrefix + '" do not seem to be present in the directory "' + trainedCkptPrefixPath + '"')
        print('was the training completed successfully?')

    return True


#######################################################################################################################
if __name__ == '__main__':
    tf.app.run()
