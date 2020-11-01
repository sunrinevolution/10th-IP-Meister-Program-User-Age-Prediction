from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, Activation, \
                                    BatchNormalization, Add, Reshape, DepthwiseConv2D, LeakyReLU
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

def make_divisible(v, divisor, min_val=None) -> int:
  if min_val is None:
    min_val = divisor
  new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
  if new_v < .9 * v:
    new_v += divisor
  return new_v

def relu(x):
  return K.relu(x, max_value=6.0)

def conv_block(inputs, filters, kernel, strides):
  channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

  x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
  x = BatchNormalization(axis=channel_axis)(x)
  return Activation(relu)(x)

def bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
  channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
  tchannel = K.int_shape(inputs)[channel_axis] * t
  cchannel = int(filters * alpha)

  x = conv_block(inputs, tchannel, (1, 1), (1, 1))

  x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
  x = BatchNormalization(axis=channel_axis)(x)
  x = Activation(relu)(x)

  x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
  x = BatchNormalization(axis=channel_axis)(x)

  if r:
    x = Add()([x, inputs])

  return x

def inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
  x = bottleneck(inputs, filters, kernel, t, alpha, strides)

  for i in range(1, n):
    x = bottleneck(x, filters, kernel, t, alpha, 1, True)
  
  return x

def MobileNetv2(input_shape, k, alpha=1.0):
  inputs = Input(shape=input_shape)

  first_filters = make_divisible(32 * alpha, 8)
  x = conv_block(inputs, first_filters, (3, 3), strides=(2, 2))

  x = inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
  x = inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
  x = inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
  x = inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
  x = inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
  x = inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
  x = inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

  if alpha > 1.0:
    last_filters = make_divisible(1280 * alpha, 8)
  else:
    last_filters = 1280
  
  x = conv_block(x, last_filters, (1, 1), strides=(1, 1))
  x = GlobalAveragePooling2D()(x)
  x = Reshape((1, 1, last_filters))(x)
  x = Dropout(0.3, name='Dropout')(x)
  x = Conv2D(k, (1, 1), padding='same')(x)

  x = Activation('softmax', name='softmax')(x)
  output = Reshape((k,))(x)

  model = Model(inputs, output)
  plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

  return model

if __name__ == '__main__':
  model = MobileNetv2((224, 224, 3), 100, 1.0)
  print(model.summary())