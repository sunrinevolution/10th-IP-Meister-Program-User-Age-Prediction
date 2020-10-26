import os
import sys
import cv2
import argparse
import pandas as pd
from MobileNetv2 import MobileNetv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Reshape, Activation
from tensorflow.keras.models import Model

def generate(batch, size, train_path, valid_path):
  train_file = os.path.join('./data', train_path)
  train_df = pd.read_csv(train_file)
  train_df.rename(columns={'image':'filename', 'label':'class'}, inplace=True)

  vaild_file = os.path.join('./data', valid_path)
  vaild_df = pd.read_csv(vaild_file)
  vaild_df.rename(columns={'image':'filename', 'label':'class'}, inplace=True)

  train_df["class"] = train_df["class"].astype(str)
  vaild_df["class"] = vaild_df["class"].astype(str)

  train_gen = ImageDataGenerator(
      rescale=1. / 255,
      shear_range=0.2,
      zoom_range=0.2,
      rotation_range=90,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True
    )
  vaild_gen = ImageDataGenerator(rescale=1. / 255)

  train_generator = train_gen.flow_from_dataframe(
      dataframe=train_df,
      directory='./data/',
      x_col='filename',
      y_col='class',
      subset='training',
      # classes=labels,
      target_size=[size, size],
      batch_size=batch,
      class_mode='categorical'
    )
  print(train_generator.class_indices)
  validation_generator = vaild_gen.flow_from_dataframe(
      dataframe=vaild_df,
      directory='./data/',
      x_col='filename',
      y_col='class',
      subset='training',
      # classes=labels,
      target_size=[size, size],
      batch_size=batch,
      class_mode='categorical'
    )
  
  train_shape = train_df.shape[0]
  vaild_shape = vaild_df.shape[0]

  return train_generator, validation_generator, train_shape, vaild_shape

def fine_tune(num_classes, weights, model):
  model.load_weights(weights)

  x = model.get_layer('Dropout').output
  x = Conv2D(num_classes, (1, 1), padding='same')(x)
  x = Activation('softmax', name='softmax')(x)
  output = Reshape((num_classes,))(x)

  model = Model(inputs=model.input, outputs=output)
  return model

def train(batch, epochs, num_classes, size, weights, tclasses, train_path, valid_path):
  train_generator, validation_generator, count1, count2 = generate(batch, size, train_path, valid_path)
  
  if weights:
    model = MobileNetv2((size, size, 3), tclasses)
    model = fine_tune(num_classes, weights, model)
  else:
    model = MobileNetv2((size, size, 3), num_classes)
  
  optimizer = Adam()
  earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  hist = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=count1 // batch,
    validation_steps=count2 // batch,
    epochs=epochs,
    callbacks=[earlystop]
  )

  if not os.path.exists('model'):
    os.makedirs('model')

  df = pd.DataFrame.from_dict(hist.history)
  df.to_csv('model/hist.csv', encoding='utf-8', index=False)
  model.save_weights('model/weights_'+train_path.split('.')[0]+'.h5')

if __name__ == '__main__':
  train(32, 10, 79, 224, False, 0, 'train.txt', 'validation.txt')