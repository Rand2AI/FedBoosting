from abc import ABC
from keras import backend as K
from keras.layers import Flatten, Dense, Input, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Permute, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.optimizers import Adam, SGD, Adadelta
import tensorflow as tf

def multi_gpu_model__(model, gpus):
  if isinstance(gpus, (list, tuple)):
    num_gpus = len(gpus)
    target_gpu_ids = gpus
  else:
    num_gpus = gpus
    target_gpu_ids = range(num_gpus)

  def get_slice(data, i, parts):
    shape = tf.shape(data)
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == num_gpus - 1:
      size = batch_size - step * i
    else:
      size = step
    size = tf.concat([size, input_shape], axis=0)
    stride = tf.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return tf.slice(data, start, size)

  all_outputs = []
  for i in range(len(model.outputs)):
    all_outputs.append([])

  # Place a copy of the model on each GPU,
  # each getting a slice of the inputs.
  for i, gpu_id in enumerate(target_gpu_ids):
    with tf.device('/gpu:%d' % gpu_id):
      with tf.name_scope('replica_%d' % gpu_id):
        inputs = []
        # Retrieve a slice of the input.
        for x in model.inputs:
          input_shape = tuple(x.get_shape().as_list())[1:]
          slice_i = Lambda(get_slice,
                           output_shape=input_shape,
                           arguments={'i': i,
                                      'parts': num_gpus})(x)
          inputs.append(slice_i)

        # Apply model on slice
        # (creating a model replica on the target device).
        outputs = model(inputs)
        if not isinstance(outputs, list):
          outputs = [outputs]

        # Save the outputs for merging back together later.
        for o in range(len(outputs)):
          all_outputs[o].append(outputs[o])

  # Merge outputs on CPU.
  with tf.device('/cpu:0'):
    merged = []
    for name, outputs in zip(model.output_names, all_outputs):
      merged.append(concatenate(outputs,axis=0, name=name))
    return Model(model.inputs, merged)

class Network(ABC):
    """
    Network Architecture ofround_mean_loss CRNN Model
    """
    def __init__(self, config=None):
        # Define class properties
        self.gpuIDs = None
        self.config = config
        self.images = None
        self.labels = None
        self.tPreds = None
        self.losses = None
        self.inList = None
        self.stemNet = None
        self.graphsModel = None
        self.deviceModel = None
        if self.config['DEVICE']['DEVICE_TOUSE'] == 'CPU':
            self.parallel = False
        else:
            if isinstance(self.config['DEVICE']['DEVICE_GPUID'], list) and len(self.config['DEVICE']['DEVICE_GPUID']) >= 2:
                print(">>>>> parallel training")
                self.parallel = True
                self.gpuIDs = self.config['DEVICE']['DEVICE_GPUID']
                # self.gpuIDs = [x for x in range(len(self.config['DEVICE']['DEVICE_GPUID']))]
            else:
                self.parallel = False
        self.backbone()
        self.framework()

    def framework(self):
        if self.config['WORKMODE'] == 'Estimation':
            self.preEstimation()
        elif self.config['WORKMODE'] == 'Inference':
            self.preInference()
        else:
            raise ValueError('No Such Work Mode: ' + self.config['WORKMODE'])

    def backbone(self):
        width = self.config['NETWORK']['IMAGE_WIDTH'] if self.config['NETWORK']['IMAGE_WIDTH'] > 0 else None
        self.images = Input(name='the_image', shape=(self.config['NETWORK']['IMAGE_HEIGHT'], width, self.config['NETWORK']['IMAGE_CHANNEL']))

        if self.config['NETWORK']['BACKBONE_CNN_FLOW'] == 'Original':
            self.stemNet = Network.cnn_backbone_Original(self.images)
        else:
            raise ValueError('No Such Backbone: ' + self.config['NETWORK']['BACKBONE_CNN_FLOW'])

        self.tPreds = Network.rnn_backbone(self.stemNet.output,
                                           tCell=self.config['NETWORK']['BACKBONE_RNN_CELL'],
                                           nCell1=self.config['NETWORK']['BACKBONE_RNN1_UNIT'],
                                           nChar=self.config['NETWORK']['DICTIONARY_SIZE'])

    def preEstimation(self):
        label_texts = Input(name='the_label_text', shape=[self.config['NETWORK']['MAX_LABEL_LENGTH']], dtype='float32')
        length_texts = Input(name='the_length_texts', shape=[1], dtype='int32')
        length_image = Input(name='the_length_image', shape=[1], dtype='int32')
        self.labels = [label_texts, length_image, length_texts]
        self.inList = [self.images] + self.labels

        self.losses = Lambda(Network.ctc_cost_lambda_func, name='loss_ctc')([label_texts, self.tPreds, length_image, length_texts])
        self.graphsModel = Model(inputs=self.inList, outputs=self.losses)
        if self.config['ESTIMATION']['STEMFREEZE']:
            for layer in self.stemNet.layers:
                layer.trainable = False
            self.stemNet.trainable = False
        if self.config['ESTIMATION']['OPT_BACKPROP'] == 'ADAM':
            opt = Adam(self.config['ESTIMATION']['LEARNING_RATE'])
        elif self.config['ESTIMATION']['OPT_BACKPROP'] == 'SGD':
            opt = SGD(self.config['ESTIMATION']['LEARNING_RATE'], self.config['ESTIMATION']['SGD_MOMENTUM'],
                      self.config['ESTIMATION']['SGD_DECAY'], self.config['ESTIMATION']['SGD_NESTEROV'])
        elif self.config['ESTIMATION']['OPT_BACKPROP'] == 'rmsprop':
            opt = 'rmsprop'
        else:
            opt = Adadelta()
        if self.parallel:
            self.deviceModel = multi_gpu_model(self.graphsModel, gpus=self.gpuIDs, cpu_merge=True, cpu_relocation=True)
        elif not self.parallel:
            self.deviceModel = self.graphsModel
        self.deviceModel.compile(opt, loss={'loss_ctc': lambda y_true, y_pred: y_pred}, metrics=['accuracy'])
        self.deviceModel.summary()

    def preInference(self):
        self.graphsModel = Model(inputs=self.images, outputs=self.tPreds)
        try:
            self.deviceModel = multi_gpu_model__(self.graphsModel, gpus=self.gpuIDs) # gpuIDs must be the same with training set
        except:
            self.deviceModel = self.graphsModel
        self.deviceModel.summary()
        if self.config['INFERENCE']['WEIGHT_TOLOAD'] != "":
            self.deviceModel.load_weights(self.config['INFERENCE']['WEIGHT_TOLOAD'])
        else:
            raise Exception(sys.exc_info())

    @staticmethod
    def ctc_cost_lambda_func(args):
        y_true, y_pred, input_length, label_length = args
        return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    @staticmethod
    def cnn_backbone_Original(inTensor=None):
        if inTensor is None:
            raise TypeError('No Input')
        # CNN Block 1 with 64-Conv2D Kernels
        m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1')(inTensor)
        m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool1')(m)
        # CNN Block 2 with 128-Conv2D Kernels
        m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1')(m)
        m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool1')(m)
        # CNN Block 3 with 2 256-Conv2D Kernels
        m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv1')(m)
        m = BatchNormalization(axis=3, name='block3_bn1')(m)
        m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv2')(m)
        m = ZeroPadding2D(padding=((0, 0), (0, 1)), name='block3_zp2')(m)
        m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='block3_pool2')(m)
        # CNN Block 4 with 2 512-Conv2D Kernels
        m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv1')(m)
        m = BatchNormalization(axis=3, name='block4_bn1')(m)
        m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv2')(m)
        m = ZeroPadding2D(padding=((0, 0), (1, 1)), name='block4_zp2')(m)
        m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='block4_pool2')(m)
        # CNN Block 5 with 1 512-Conv2D Kernels
        m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='block5_conv1')(m)
        outTensor = BatchNormalization(axis=3, name='block5_bn1')(m)

        model = Model(inTensor, outTensor, name='orignial')
        return model

    @staticmethod
    def rnn_backbone(inTensor=None, tCell='GRU', nCell1=None, nChar=1000):
        # RNN Block 1 with Bidirectional Model for Transcription
        # Map-to-Sequence by swapping height and width
        if nCell1 is None:
            nCell1 = [256, 256]
        m = Permute((2,1,3), name='rnn1_permute')(inTensor)
        # Flatten the feature along width
        m = TimeDistributed(Flatten(), name='rnn1_timedistrib')(m)
        nStack = 1
        if tCell == 'LSTM':
            for nUnits in nCell1:
                m = Bidirectional(LSTM(nUnits, return_sequences=True, implementation=2), name='rnn1_blstm'+ str(nStack))(m)
                nStack += 1
        elif tCell == 'GRU':
            for nUnits in nCell1:
                m = Bidirectional(GRU(nUnits, return_sequences=True, implementation=2), name='rnn1_bgru'+ str(nStack))(m)
                nStack += 1
        else:
            raise ValueError('No Such Cell')
        outTensor1 = Dense(nChar, name='rnn1_out', activation='softmax')(m)

        return outTensor1

if __name__ =="__main__":
    import FLutils, os
    CRNNconfig = FLutils.get_config(os.path.dirname(os.path.realpath(__file__)), 'Estimation')
    net = Network(CRNNconfig)
    print(net.deviceModel.metrics_names)