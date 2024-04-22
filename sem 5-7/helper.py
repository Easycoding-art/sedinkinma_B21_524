import math
import numpy as np
from PIL import Image

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

def T(image_arr):
    bins = np.arange(np.min(image_arr) - 1, np.max(image_arr) + 1)
    hist, base = np.histogram(image_arr, bins=bins, density=True)
    base = base[1:].astype(np.uint8)
    w_0 = np.cumsum(hist)
    w_1 = np.ones(shape=w_0.shape) - w_0
    t_rank = 0
    i_max = 0
    for i, (w0, w1) in enumerate(zip(w_0, w_1)):
        m_0 = np.sum(base[:i] * hist[:i] / w0)
        m_1 = np.sum(base[i + 1:] * hist[i + 1:] / w1)
        d_0 = np.sum(hist[:i] * (base[:i] - m_0)**2)
        d_1 = np.sum(hist[i + 1:] * (base[i + 1:] - m_1)**2)
        d_all = w0 * d_0 + w1 * d_1
        d_class = w0 * w1 * (m_0 - m_1)**2
        if d_all == 0:
            i_max = i
            break
        if d_class / d_all > t_rank:
            t_rank = d_class / d_all
            i_max = i

    return base[i_max]

def Otsu_binarization(image) :
  img = semitone(image)
  image_arr = np.array(img)
  t = T(image_arr)
  image_arr[image_arr > t] = 255
  image_arr[image_arr <= t] = 0
  new_image = Image.fromarray(image_arr.astype(np.uint8), 'L')
  return new_image

def get_diff(original, recognized) :
    diff = len(recognized) - len(original)
    k = 0
    for i in range(len(original)) :
        if original[i] == recognized[i] :
            k+=1
    return diff + k

def Manhatten(arr1, arr2) :
    if len(arr1) != len(arr2) :
        raise ValueError('Length of arrays must be same!')
    p = 0
    for i in range(len(arr1)) :
        p += max(-(arr2[i] - arr2[i]), arr2[i] - arr2[i])
    return p