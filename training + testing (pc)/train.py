import functools
import json
import os
import tensorflow as tf

from object_detection import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util

# module-level variables ##############################################################################################
# lokasi file file yang dibutuhkan dan lokasi hasil training akan disimpan
PIPELINE_CONFIG_PATH = os.getcwd() + "/" + "ssd_mobilenet_v2_coco.config"

MODEL_DIR = os.getcwd() + "/" + "ssd_mobilenet_v2_coco_2018_03_29"

FILES_MODEL_DIR_MUST_CONTAIN = [ "checkpoint" ,
                                 "frozen_inference_graph.pb",
                                 "model.ckpt.data-00000-of-00001",
                                 "model.ckpt.index",
                                 "model.ckpt.meta"]

TRAINING_DATA_DIR = os.getcwd() + "/training_data/"

NUM_CLONES = 1

CLONE_ON_CPU = False

#######################################################################################################################
def main(_):
    print("starting program . . .")

    tf.logging.set_verbosity(tf.logging.INFO)

    if not checkIfNecessaryPathsAndFilesExist():
        return
    
    configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG_PATH)
    tf.gfile.Copy(PIPELINE_CONFIG_PATH, os.path.join(TRAINING_DATA_DIR, 'pipeline.config'), overwrite=True)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=True)

    def get_next(config):
        return dataset_builder.make_initializable_iterator(dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
        worker_replicas = len(cluster_data['worker']) + 1

    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc', job_name=task_info.type, task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    trainer.train(create_input_dict_fn, model_fn, train_config, master, task, NUM_CLONES, worker_replicas,
                  CLONE_ON_CPU, ps_tasks, worker_job_name, is_chief, TRAINING_DATA_DIR)

#######################################################################################################################
# melakukan pengecekan file-file yang dibutuhkan
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(PIPELINE_CONFIG_PATH):
        print('ERROR: the big (pipeline).config file "' + PIPELINE_CONFIG_PATH + '" does not seem to exist')
        return False

    missingModelMessage = "Did you download and extract the model from the TensorFlow GitHub models repository detection model zoo?" + "\n" + \
                          "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md"

    if not os.path.exists(MODEL_DIR):
        print('ERROR: the model directory "' + MODEL_DIR + '" does not seem to exist')
        print(missingModelMessage)
        return False

    for necessaryModelFileName in FILES_MODEL_DIR_MUST_CONTAIN:
        if not os.path.exists(os.path.join(MODEL_DIR, necessaryModelFileName)):
            print('ERROR: the model file "' + MODEL_DIR + "/" + necessaryModelFileName + '" does not seem to exist')
            print(missingModelMessage)
            return False

    if not os.path.exists(TRAINING_DATA_DIR):
        print('ERROR: TRAINING_DATA_DIR "' + TRAINING_DATA_DIR + '" does not seem to exist')
        return False

    return True

#######################################################################################################################
if __name__ == '__main__':
    tf.app.run()
