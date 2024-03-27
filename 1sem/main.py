from PIL import Image
import numpy as np
import os

def KNN(image, k) :
  src = image.convert('RGB')
  image_arr = np.array(src)
  width, height = image.size
  new_width = round(k * width)
  new_height = round(k * height)
  new_image_arr = np.zeros(shape=(new_height, new_width, 3))
  for x in range(new_width):
      for y in range(new_height):
          src_x = min(int(round(float(x) / float(new_width) * float(width))), width - 1)
          src_y = min(int(round(float(y) / float(new_height) * float(height))), height - 1)
          new_image_arr[y, x] = image_arr[src_y, src_x]
  new_image = Image.fromarray(new_image_arr.astype(np.uint8), 'RGB')
  return new_image

#tuple (3, 2) mean 3/2, 2 steps
def test(*args) :
  current_dir = os.path.dirname(os.path.abspath(__file__))
  output_path = os.path.join(current_dir, 'output')
  input_path = os.path.join(current_dir, 'input')
  for arg in args :
     os.makedirs(f'{output_path}/K = {arg}')
  file_list = os.listdir(input_path)
  for file in file_list :
    image = Image.open(f'{input_path}/{file}')
    for arg in args :
        if isinstance(arg, tuple) :
            img = KNN(image, 1/arg[1])
            img2 = KNN(img, arg[0])
            img2.save(f'{output_path}/K = {arg}/{file}')
        elif isinstance(arg, int) or isinstance(arg, float) :
            img = KNN(image, arg)
            img.save(f'{output_path}/K = {arg}/{file}')

if __name__ == '__main__' :
   test(0.4, 2, (3, 2))