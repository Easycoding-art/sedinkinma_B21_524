from PIL import Image
import numpy as np
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

def test() :
  current_dir = os.path.dirname(os.path.abspath(__file__))
  output_path = os.path.join(current_dir, 'output')
  input_path = os.path.join(current_dir, 'input')
  file_list = os.listdir(input_path)
  for file in file_list :
    image = Image.open(f'{input_path}/{file}')
    img2 = semitone(image)
    img2.save(f'{output_path}/{file}')

if __name__ == '__main__' :
   test()