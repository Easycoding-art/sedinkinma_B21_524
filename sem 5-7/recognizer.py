import pandas as pd
import helper
import numpy as np
from PIL import Image
import os
import sys
def check_cohesion(image_arr, index, neighbor_index):
    high_level = image_arr[0:int(0.3*image_arr.shape[0]), :]
    mid_level = image_arr[int(0.3*image_arr.shape[0]):int(0.7*image_arr.shape[0]), :]
    low_level = image_arr[int(0.7*image_arr.shape[0]):image_arr.shape[0], :]
    columns = []
    columns_max = []
    high_level_max = []
    mid_level_max = []
    low_level_max = []
    for i in range(image_arr.shape[1]) :
        columns.append(sum(image_arr[:, i])/image_arr.shape[1])
        columns_max.append(max(image_arr[:, i]))
        high_level_max.append(max(high_level[:, i]))
        mid_level_max.append(max(mid_level[:, i]))
        low_level_max.append(max(low_level[:, i]))
    if (high_level_max[index] == high_level_max[neighbor_index]
        and mid_level_max[index] == mid_level_max[neighbor_index]
        and low_level_max[index] == mid_level_max[neighbor_index]) :
        requirement_1 = True
    else :
        requirement_1 = False
    if (columns[index] < columns_max[neighbor_index]) :
        requirement_2 = True
    else :
        requirement_2 = False
    if (columns_max[index] > 2*abs(columns_max[index] - columns_max[neighbor_index])) :
        requirement_3 = True
    else :
        requirement_3 = False
    return requirement_1 and requirement_2 and requirement_3

def characterics(image) :
    image_arr = np.array(image)
    width, height = image.size
    new_image_arr = np.zeros(shape=(height, width))
    for x in range(width):
        for y in range(height):
            new_image_arr[y, x] = (image_arr[y, x, 0] + image_arr[y, x, 1] + image_arr[y, x, 2])/3
    new_arr = np.zeros(new_image_arr.shape)
    new_arr[new_image_arr != 255] = 1
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
            name = symbol[0].replace('.png', '')
            values = symbol[1:]
            p = helper.Manhatten(values, stats)
            classes.append(name)
            chances.append(p)
        index = chances.index(max(chances))
        word = word + classes[index]
    return word
def split_picture(image) :
    image_arr = np.array(image)
    m = image_arr.shape[1]
    n = image_arr.shape[0]
    img = image.copy()
    copy_arr = np.array(img)
    for i in range(m) :
        for j in range(n) :
            if image_arr[j][i] == 255 :
                if j+1 < n:
                    if i +1 < m:
                        copy_arr[j+1][i+1] = 255
                    copy_arr[j+1][i-1] = 255
                    copy_arr[j+1][i] = 255
                copy_arr[j-1][i-1] = 255
                if i + 1 < m:
                    copy_arr[j-1][i+1] = 255
                    copy_arr[j][i+1] = 255
                copy_arr[j-1][i] = 255
                copy_arr[j][i-1] = 255
    columns = []
    l = 0
    r = 0
    sumbol_cord = []
    for i in range(m) :
        columns.append(sum(copy_arr[:, i])/m)
    c = sum(columns)/len(columns)
    c_l = 0.01*c
    c_r = 0.01*c
    for i in range(len(columns)) :
        if i == 0 :
            left = 0
        else :
            left = columns[i - 1]
        if left < c_l and columns[i] > c_l and columns[i+1] > c_l and l == 0 :
            l = i
        if (columns[i-2] > c_r
            and columns[i-1] > c_r 
            and columns[i] < c_r 
            and columns[i+1] < c_r 
            and columns[i+2] < c_r 
            and columns[i+3] < c_r 
            and columns[i+4] < c_r 
            and l != 0) :
            r = i
            sumbol_cord.append((l, r))
            l = 0
    result = []
    for s in sumbol_cord :
        new = Image.fromarray(image_arr[:, s[0]:s[1]].astype(np.uint8), 'L')
        result.append(new)
    return result

def split_word(image) :
    width, height = image.size
    image_arr = np.array(image)
    #Находим потенциальные границы
    d_j = 0.3 * height
    columns = []
    for i in range(width) :
        columns.append(sum(image_arr[:, i])/width)
    indexes = []
    start = 0
    end = int(d_j)
    while True :
        index = columns[start:end+1].index(min(columns[start:end+1]))
        indexes.append(index + start)
        start+=index
        end+=index
        if end >= len(columns) :
            break
    #Первичное отсеивание ложных границ
    c = sum(columns)/len(columns)
    c_b = 0.01*c
    filtered_indexes = list(filter(lambda x: columns[x] < c_b and
                                   (columns[x+2] > c_b or columns[x-2] > c_b), indexes))
    #Вторичное отсеивание ложных границ
    result_indexes = []
    d_min= 0.4 * height
    for i in len(filtered_indexes) :
        cohesion_right = check_cohesion(image_arr, filtered_indexes[i], filtered_indexes[i]+1)
        cohesion_left = check_cohesion(image_arr, filtered_indexes[i], filtered_indexes[i]-1)
        if cohesion_left and cohesion_right == False :
            d_k = filtered_indexes[i] - filtered_indexes[i-1]
            if d_k > d_min :
                result_indexes.append(filtered_indexes[i])
    #Собираем буквы для классификации
    image_arr[image_arr == 255] = 1
    image_arr[image_arr == 0] = 255
    image_arr[image_arr == 1] = 0
    result = []
    for s in range(0, len(result_indexes), 2) :
        new = Image.fromarray(image_arr[:, result_indexes[s]:s[s+1]].astype(np.uint8), 'L')
        new.show()
        result.append(new)
    return result

def recognize(file_path, text, symbol_file) :
    arr = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    df = pd.read_csv(symbol_file)
    if df.shape[0] >= 10 :
            splitter = arr[df.shape[0]-10]
    else :
        splitter = df.shape[0]
    image = Image.open(file_path)
    img = helper.Otsu_binarization(image)
    word_pics = split_picture(img)
    recognized_text = ''
    for im in word_pics :
         letters = split_word(im)
         word = get_word(letters, symbol_file)
         recognized_text = recognized_text + word + splitter
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