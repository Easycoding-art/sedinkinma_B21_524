import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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
    len_diff = abs(len(recognized) - len(original))
    if len(recognized) > len(original) :
        original = original + '#'*len_diff
    else :
        recognized = recognized + '#'*len_diff
    k = 0
    for i in range(len(original)) :
        if original[i] != recognized[i] :
            k+=1
    return k

def Evclid(arr1, arr2) :
    if len(arr1) != len(arr2) :
        raise ValueError('Length of arrays must be same!')
    p = 0
    for i in range(len(arr1)) :
        p += (arr1[i] - arr2[i])**2
    return math.sqrt(p)

def get_profiles(image):
    image_arr = np.array(image)
    image_arr[image_arr == 0] = 1
    image_arr[image_arr == 0] = 0
    w, h = image_arr.shape
    return {
        'x': {
            'y': np.sum(image_arr, axis=0),
            'x': np.arange(start=1, stop=h + 1).astype(int)
        },
        'y': {
            'y': np.arange(start=1, stop=w + 1).astype(int),
            'x': np.sum(image_arr, axis=1)
        }
    }

def save_profiles(img, name, file_path):
    profiles = get_profiles(img)
    fig, axs = plt.subplots(nrows= 2 , ncols= 1 )
    #add data to plots
    axs[0].bar(x=profiles['x']['x'], height=profiles['x']['y'], width=0.9)
    axs[0].axis(xmin=0,xmax=90, ymin=0,ymax=90)
    #axs[0].ylim(0, 90)
    #axs[0].xlim(0, 90)
    #axs[0].subtitle('X')
    axs[1].barh(y=profiles['y']['y'], width=profiles['y']['x'], height=0.9)
    axs[1].axis(xmin=0,xmax=90, ymin=90,ymax=0)
    #axs[1].ylim(90, 0)
    #axs[1].xlim(0, 90)
    #axs[1].subtitle('Y')
    plt.savefig(f'{file_path}/{name}.png')
    plt.clf()

def calculate_profiles(img):
    image_arr = np.array(img)
    image_arr[image_arr == 0] = 1
    image_arr[image_arr == 0] = 0
    profile_x = np.sum(img, axis=0)
    profile_y = np.sum(img, axis=1)
    return {
        'x': profile_x,
        'y': profile_y
    }

def split_colors(image) :
    image_arr = np.array(image)
    image_arr[image_arr == 0] = 1
    image_arr[image_arr == 255] = 0
    image_arr[image_arr == 1] = 255
    return Image.fromarray(image_arr.astype(np.uint8), 'L')

def make_normal(image) :
    image_arr = np.array(image)
    if image_arr[0, 0] == 255 :
        return split_colors(image)
    else :
        return image