from PIL import Image
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
image = Image.open(f'{current_dir}/japan.png')
