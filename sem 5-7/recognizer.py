import pandas as pd
import helper
import numpy as np
from PIL import Image

def characterics(image) :
    width, height = image.size
    new_image_arr = np.zeros(shape=(height, width))
    image_arr = np.array(image)
    new_arr = np.zeros(new_image_arr.shape)
    new_arr[image_arr != 255] = 1
    #weight
    weight = new_arr.sum()
    normalized_weight = weight / (new_image_arr.shape[0] * new_image_arr.shape[1])
    x_avg = 0
    for i in range(new_image_arr.shape[0]) :
        for j in range(new_image_arr.shape[1]) :
            x_avg = i*new_arr[i, j]
    x_avg = x_avg / weight
    x_avg_rel = (x_avg - 1)/(new_image_arr.shape[1] - 1)
    y_avg = 0
    for i in range(new_image_arr.shape[0]) :
        for j in range(new_image_arr.shape[1]) :
            y_avg = j*new_arr[i, j]
    y_avg = y_avg / weight
    y_avg_rel = (y_avg - 1)/(new_image_arr.shape[0] - 1)
    moment_x = 0
    for i in range(new_image_arr.shape[0]) :
        for j in range(new_image_arr.shape[1]) :
            moment_x = ((j - y_avg)**2)*new_arr[i, j]
    moment_y = 0
    for i in range(new_image_arr.shape[0]) :
        for j in range(new_image_arr.shape[1]) :
            moment_y = ((i - x_avg)**2)*new_arr[i, j]
    moment_x_rel = moment_x/(new_image_arr.shape[0]**2 * new_image_arr.shape[1]**2)
    moment_y_rel = moment_y/(new_image_arr.shape[0]**2 * new_image_arr.shape[1]**2)
    moment_45 = 0
    for i in range(new_image_arr.shape[0]) :
        for j in range(new_image_arr.shape[1]) :
            moment_45 = ((j - y_avg - i + x_avg)**2)*new_arr[i, j]
    moment_135 = 0
    for i in range(new_image_arr.shape[0]) :
        for j in range(new_image_arr.shape[1]) :
            moment_135 = ((j - y_avg + i - x_avg)**2)*new_arr[i, j]
    moment_45_rel = moment_45/(new_image_arr.shape[0]**2 * new_image_arr.shape[1]**2)
    moment_135_rel = moment_135/(new_image_arr.shape[0]**2 * new_image_arr.shape[1]**2)
    return [normalized_weight, x_avg_rel, y_avg_rel, 
            moment_x_rel, moment_y_rel, moment_45_rel, moment_135_rel]

def get_text_box(img, h_gap, v_gap):
    profiles = helper.calculate_profiles(img)
    x1, x2, y1, y2 = None, None, None, None
    i = 0
    while i < profiles['x'].shape[0]:
        current = profiles['x'][i]
        if current != 0 and x1 == None:
            x1 = i
        elif current == 0:
            if x1 == None:
                pass
            else:
                count = 0
                while profiles['x'][i + count] == 0:
                    if count == h_gap:
                        x2 = i
                        i = profiles['x'].shape[0]
                        break
                    if i + count >= profiles['x'].shape[0] - 1:
                        x2 = i
                        i = profiles['x'].shape[0]
                        break
                    count += 1
                i += count
                continue
        i += 1
    if x2 == None:
        x2 = i

    i = 0
    while i < profiles['y'].shape[0]:
        current = profiles['y'][i]
        if current != 0 and y1 == None:
            y1 = i
        elif current == 0:
            if y1 == None:
                pass
            else:
                count = 0
                while profiles['y'][i + count] == 0:
                    if count == v_gap:
                        y2 = i
                        count += 1
                        break
                    if i + count >= profiles['y'].shape[0] - 1:
                        y2 = i
                        count += 1
                        break
                    count += 1
                i += count
                continue
        i += 1
    if y2 == None:
        y2 = i
    image_arr = np.array(img)
    return Image.fromarray(image_arr[y1:y2, x1:x2].astype(np.uint8), 'L')

def get_symbol_boxes(img):
    profiles = helper.calculate_profiles(img)
    borders = []
    i = 0
    while i < profiles['x'].shape[0]:
        current = profiles['x'][i]
        if current != 0:
            x1, x2 = None, None
            x1 = i
            count = 0
            while profiles['x'][i + count] != 0:
                count += 1
            i += count
            x2 = i
            borders.append((x1, x2))
        i += 1
    result = []
    image_arr = np.array(img)
    for border in borders :
        new = Image.fromarray(image_arr[:, border[0]:border[1]].astype(np.uint8), 'L')
        result.append(new)
    return result

def get_word(images, symbol_file) :
    df = pd.read_csv(symbol_file)
    df = df.drop(['weight','x_avg', 'y_avg', 
                  'moment_x', 'moment_y', 
                  'moment_45', 'moment_135'], axis=1)
    symbols = df.values
    word = ''
    for image in images :
        stats = characterics(image)
        classes = []
        chances = []
        for symbol in symbols :
            name = symbol[1].replace('.png', '')
            values = symbol[2:]
            p = helper.Evclid(values, stats)
            classes.append(name)
            chances.append(p)
        index = chances.index(min(chances))
        word = word + classes[index]
    return word

def recognize(file_path, text, symbol_file) :
    img = Image.open(file_path)
    img2 = helper.Otsu_binarization(img)
    normal = helper.make_normal(img2)
    letters = helper.get_symbol_boxes(normal)
    recognized =[]
    for letter in letters :
        real_symbol = get_text_box(letter, 17, 90)
        recognized.append( helper.split_colors(real_symbol))
        helper.split_colors(real_symbol).save(f'results/{i}.png')
    recognized_text = get_word(recognized, symbol_file)
    diff = helper.get_diff(text, recognized_text)
    return recognized_text, diff

def test(file_path, test_path, symbol_file) :
    df = pd.read_csv(file_path)
    files = df['file'].to_list()
    texts = df['text'].to_list()
    results = []
    diffs = []
    for i in range(len(files)) :
        result, difference = recognize(f'{test_path}/{files[i]}', texts[i], symbol_file)
        results.append(result)
        diffs.append(difference)
    df.insert(loc= len(df.columns) , column='recognized text', value=results)
    df.insert(loc= len(df.columns) , column='difference', value=diffs)
    df.to_csv('test_results.csv')