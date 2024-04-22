from PIL import Image
import numpy as np
import random
from scipy import signal
import os

def semitone(image) :
  src = image.convert('RGB')
  image_arr = np.array(src)
  width, height = image.size
  new_image_arr = np.zeros(shape=(height, width))
  for x in range(width):
      for y in range(height):
          new_image_arr[y, x] = (image_arr[y, x, 0] + image_arr[y, x, 1] + image_arr[y, x, 2])/3
  new_image = Image.fromarray(new_image_arr.astype(np.uint8), 'L')
  return new_image

def get_pixel(pixel_grad, T) :
  if pixel_grad > T :
    return 255
  else :
    return 0

def Prewitt_algolitmn(image, T) :
  img = semitone(image)
  image_arr = np.array(img)
  mask_x = np.array([[-1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1]])
  mask_y = np.array([[-1, 0, 0, 0, 1],
                    [-1, 0, 0, 0, 1],
                    [-1, 0, 0, 0, 1],
                    [-1, 0, 0, 0, 1],
                    [-1, 0, 0, 0, 1]])
  #делаем свертку(порядок матриц важен!!!)
  G_x = signal.convolve2d(image_arr, mask_x, boundary='symm', mode='same')
  G_y = signal.convolve2d(image_arr, mask_y, boundary='symm', mode='same')
  #находим градиент
  grad = np.sqrt(np.square(G_x) + np.square(G_y))#возведение в квадрат поэлементное, а не матричное
  #нормализация градиента
  grad_norm = grad*(255/np.max(grad))
  for x in range(image_arr.shape[0]) :
    for y in range(image_arr.shape[1]) :
      image_arr[x, y] = get_pixel(grad_norm[x, y], T)
  new_image = Image.fromarray(image_arr.astype(np.uint8), 'L')
  return new_image

def test(*args) :
  current_dir = os.path.dirname(os.path.abspath(__file__))
  output_path = os.path.join(current_dir, 'output')
  input_path = os.path.join(current_dir, 'input')
  file_list = os.listdir(input_path)
  for arg in args :
     os.makedirs(f'{output_path}/T = {arg}')
  for file in file_list :
    for a in args :
        image = Image.open(f'{input_path}/{file}')
        img2 = Prewitt_algolitmn(image, a)
        img2.save(f'{output_path}/T = {a}/{file}')

if __name__ == '__main__' :
   test(50)