import os
import cv2
from tqdm import tqdm

train_list = []
validation_list = []
train_text = "image,label\n"
validation_text = "image,label\n"

with open('./image sets/train.txt', 'r') as f:
  train_list = f.readlines()
  for index, item in enumerate(train_list):
    train_list[index] = item.split()[0]

with open('./image sets/val.txt', 'r') as f:
  validation_list = f.readlines()
  for index, item in enumerate(validation_list):
    validation_list[index] = item.split()[0]

try:
  os.mkdir('face')
  os.mkdir('./face/train')
  os.mkdir('./face/validation')
except:
  pass

print("Train Dataset PreProcessing Start")
for index, item in tqdm(enumerate(train_list)):
  dir = item.split('A')[1].split('.')[0]
  try:
    os.mkdir(f'./face/train/{dir[0]}0')
  except:
    pass
  image_read = cv2.imread(f'./aglined faces/{item}')
  image = cv2.resize(image_read, (224, 224), interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(f'./face/train/{dir[0]}0/{item}', image)
  train_text += f'./face/train/{dir[0]}0/{item},{dir[0]}0\n'

print("Validation Dataset PreProcessing Start")
for index, item in tqdm(enumerate(validation_list)):
  dir = item.split('A')[1].split('.')[0]
  try:
    os.mkdir(f'./face/validation/{dir[0]}')
  except:
    pass
  image_read = cv2.imread(f'./aglined faces/{item}')
  image = cv2.resize(image_read, (224, 224), interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(f'./face/validation/{dir[0]}/{item}', image)
  validation_text += f'./face/validation/{dir[0]}/{item},{dir[0]}0\n'

with open('train.txt', 'a') as f:
  f.write(train_text)

with open('validation.txt', 'a') as f:
  f.write(validation_text)