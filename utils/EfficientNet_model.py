import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import copy
import math


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


DEFAULT_BLOCKS_ARGS = [{'kernel_size': 3,'repeats': 1,'filters_in': 32,'filters_out': 16,'expand_ratio': 1,'id_skip': True,'strides': 1, 'se_ratio': 0.25}, 
                       {'kernel_size': 3,'repeats': 2,'filters_in': 16,'filters_out': 24,'expand_ratio': 6,'id_skip': True, 'strides': 2,'se_ratio': 0.25}, 
                       {'kernel_size': 5,'repeats': 2,'filters_in': 24,'filters_out': 40,'expand_ratio': 6,'id_skip': True,'strides': 2,'se_ratio': 0.25},
                       {'kernel_size': 3,'repeats': 3,'filters_in': 40,'filters_out': 80,'expand_ratio': 6,'id_skip': True,'strides': 2,'se_ratio': 0.25}, 
                       {'kernel_size': 5,'repeats': 3,'filters_in': 80,'filters_out': 112,'expand_ratio': 6,'id_skip': True,'strides': 1,'se_ratio': 0.25}, 
                       {'kernel_size': 5,'repeats': 4,'filters_in': 112,'filters_out': 192,'expand_ratio': 6,'id_skip': True,'strides': 2,'se_ratio': 0.25}, 
                       {'kernel_size': 3,'repeats': 1,'filters_in': 192,'filters_out': 320,'expand_ratio': 6,'id_skip': True,'strides': 1,'se_ratio': 0.25}]


def block(inputs,
          activation=tf.nn.swish,
          drop_rate=0.,
          name='',
          filters_in=32,
          filters_out=16,
          kernel_size=3,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True):
        #Inverted Residual block
        
        filters = filters_in * expand_ratio
        bn_axis = 3
        if expand_ratio != 1:
            
            x = layers.Conv2D(
                     filters,
                            1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name=name + 'expand_conv')(inputs)
            x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
            x = layers.Activation(activation, name=name + 'expand_activation')(x)
        else:
            x = inputs
        
        x = layers.DepthwiseConv2D(
            kernel_size,
            strides = strides,
            padding = 'same',
            use_bias = False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'dwconv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
        x = layers.Activation(activation, name=name + 'activation')(x)
        
        
        # Squeeze and Excitation phase
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
            se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
            se = layers.Conv2D(
                filters_se,
                1,
                padding='same',
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + 'se_reduce')(se)
            se = layers.Conv2D(
                filters,
                    1,
                padding='same',
                activation='sigmoid',
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + 'se_expand')(se)
            x = layers.multiply([x, se], name=name + 'se_excite')

      # Output phase
        x = layers.Conv2D(
              filters_out,
              1,
              padding='same',
              use_bias=False,
              kernel_initializer=CONV_KERNEL_INITIALIZER,
              name=name + 'project_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
            x = layers.add([x, inputs], name=name + 'add')
        return x
    
    
def EfficientNet(
    width_coefficient,
    depth_coefficient,
    default_size,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    activation=tf.nn.swish,
    blocks_args=DEFAULT_BLOCKS_ARGS,
    model_name='efficientnet',
    #include_top=True,
   #weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'):
    
    if input_tensor is None:
        Input = layers.Input(shape=input_shape)
    else:
        Input = layers.Input(tensor=input_tensor, shape=input_shape)
        
    bn_axis = 3
    
    def round_filters(filters, divisor=depth_divisor):
        filters *= width_coefficient
        
        new_filters = max(divisor,int(filters+divisor/2) // divisor*divisor)
        
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)
    
    def round_repeats(repeats):
        return int(math.ceil(depth_coefficient * repeats))
    
    
    
    x = Input
    x = layers.experimental.preprocessing.Rescaling(1.0/255.0)(x)
    x = layers.experimental.preprocessing.Normalization(axis=bn_axis)(x)

    x = layers.ZeroPadding2D(
      padding=(1,1), name='stem_conv_pad')(x)
    x = layers.Conv2D(round_filters(32),
                      3,
                      strides=2,
                      padding='valid',
                      use_bias=False,
                      kernel_initializer = CONV_KERNEL_INITIALIZER, 
                      name = 'stem_conv' )(x)
    x = layers.BatchNormalization(axis=bn_axis,name='stem_bn')(x)
    x = layers.Activation(activation,name='stem_activation')(x)
    
    blocks_args = copy.deepcopy(blocks_args)
    b = 0
    blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
    for (i,args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])
            
        for j in range(round_repeats(args.pop('repeats'))):
                if j>0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = block(x,
                          activation,
                          drop_connect_rate * b /blocks,
                         name = 'block{}{}_'.format(i+1,chr(j+97)),
                         **args)
                b +=1
    
    x  = layers.Conv2D(
    round_filters(1280),
    1,
    padding = 'same',
    use_bias = False,
    kernel_initializer = CONV_KERNEL_INITIALIZER,
    name = 'top_conv')(x)
    
    x = layers.BatchNormalization(axis=bn_axis,name='top_bn')(x)
    x = layers.Activation(activation, name = 'top_activation')(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(dropout_rate,name= 'top_dropout')(x)
    x = layers.Dense(classes,
                     activation=classifier_activation,
                    kernel_initializer=DENSE_KERNEL_INITIALIZER,
                    name = 'predictions')(x)
    
    model = tf.keras.Model(Input,x,name = model_name)       
    return model    

    
# We give subscript _1 to our model to differentiate it from the model imported through tensorflow
def EfficientNetB0_1(input_tensor=None,
                   input_shape=(224,224,3),
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax'):
    return EfficientNet(
      1.0,
      1.0,
      224,
      0.2,
      model_name='efficientnetb0',
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation)


def EfficientNetB1_1(input_tensor=None,
                   input_shape=(224,224,3),
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax'):
    return EfficientNet(
      1.0,
      1.1,
      240,
      0.2,
      model_name='efficientnet-b1',
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation)


def EfficientNetB2_1(input_tensor=None,
                   input_shape=(224,224,3),
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax'):
    return EfficientNet(
      1.1,
      1.2,
      260,
      0.3,
      model_name='efficientnet-b2',
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation)


def EfficientNetB3_1(input_tensor=None,
                   input_shape=(224,224,3),
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax'):
    return EfficientNet(
      1.2,
      1.4,
      300,
      0.3,
      model_name='efficientnet-b3',
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation)


def EfficientNetB4_1(input_tensor=None,
                   input_shape=(224,224,3),
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax'):
    return EfficientNet(
      1.4,
      1.8,
      380,
      0.4,
      model_name='efficientnet-b4',
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation)

def EfficientNetB5_1(input_tensor=None,
                   input_shape=(224,224,3),
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax'):
    return EfficientNet(
      1.6,
      2.2,
      456,
      0.4,
      model_name='efficientnet-b5',
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation)

def EfficientNetB6_1(input_tensor=None,
                   input_shape=(224,224,3),
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax'):
    return EfficientNet(
      1.8,
      2.6,
      528,
      0.5,
      model_name='efficientnet-b6',
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation)

def EfficientNetB7_1(input_tensor=None,
                   input_shape=(224,224,3),
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax'):
    return EfficientNet(
      2.0,
      3.1,
      600,
      0.5,
      model_name='efficientnet-b7',
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation)

