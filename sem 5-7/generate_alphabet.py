from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np
import random
import helper

def create_alphabet(font_path, alphabet, path_name) :
    os.mkdir(path_name)
    arr = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    for i in range(len(alphabet)) :
        im = Image.new('RGB', (100,120), color=(255, 255, 255))
        # Создаем объект со шрифтом
        font = ImageFont.truetype(font_path, size=90)#D:/Roboto/Roboto-Black.ttf
        draw_text = ImageDraw.Draw(im)
        draw_text.text(
            (50, 50),
            alphabet[i],
            # Добавляем шрифт к изображению
            font=font,
            anchor="mm",
            fill=(0, 0, 0))
        binary_img = helper.Otsu_binarization(im)
        image = t.get_text_box(t.split_colors(binary_img), 17, 90)
        symbol = t.split_colors(image)
        if i >= 10 :
            name = arr[i-10]
        else :
            name = i
        symbol.save(f'{path_name}/{name}.png')

def set_tests(font_path, alphabet, test_count) :
    os.mkdir('test_cases')
    arr = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    class_dict = {alphabet[i] : (str(i) if i < 10 else arr[i-10]) for i in range(len(alphabet))}
    if len(alphabet) >= 10 :
        name = arr[len(alphabet)-10]
    else :
        name = len(alphabet)
    class_dict.update({' ' : name})
    tests = []
    language = alphabet.copy()
    language.append(' ')
    for j in range(test_count) :
        #size = random.randint(10, 90)
        size = 90
        word_len = random.randint(0, 40)
        word = ''
        for k in range(word_len) :
            word = word + random.choice(language)
        im = Image.new('RGB', (50 * word_len, size*2), 
                       color=(random.randint(0, 255),
                              random.randint(0, 255), 
                              random.randint(0, 255)))
        # Создаем объект со шрифтом
        font = ImageFont.truetype(font_path, size=size)#D:/Roboto/Roboto-Black.ttf
        draw_text = ImageDraw.Draw(im)
        draw_text.text(
            (25 * word_len, size),
            word,
            # Добавляем шрифт к изображению
            font=font,
            anchor="mm",
            fill=(random.randint(0, 255),
                  random.randint(0, 255),
                  random.randint(0, 255)))
        im.save(f'test_cases/{j}.png')
        text_to_class = [str(class_dict.get(letter)) for letter in list(word)]
        tests.append(''.join(text_to_class))
    indexes = [i for i in range(len(tests))]
    files = os.listdir('test_cases')
    df = pd.DataFrame(list(zip(files, tests)), indexes, ['file', 'text'])
    df.to_csv('test_cases.csv')

def get_characterics(file_path) :
    #file
    files = os.listdir(file_path)
    weights = []
    normalized_weights = []
    x_avgs = []
    y_avgs = []
    x_avg_rels = []
    y_avg_rels = []
    moment_x_s = []
    moment_y_s = []
    moment_x_rels = []
    moment_y_rels = []
    moment_45_s = []
    moment_135_s = []
    moment_45_rels = []
    moment_135_rels = []
    os.mkdir('profiles')
    for file in files :
        img = Image.open(f'{file_path}/{file}')
        image_arr = np.array(img)
        width, height = img.size
        new_image_arr = np.zeros(shape=(height, width))
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
        #профили
        name = file.replace('.png', '')
        helper.save_profiles(img, name, 'profiles')
        #adding
        weights.append(weight)
        normalized_weights.append(normalized_weight)
        x_avgs.append(x_avg)
        y_avgs.append(y_avg)
        x_avg_rels.append(x_avg_rel)
        y_avg_rels.append(y_avg_rel)
        moment_x_s.append(moment_x)
        moment_y_s.append(moment_y)
        moment_x_rels.append(moment_x_rel)
        moment_y_rels.append(moment_y_rel)
        moment_45_s.append(moment_45)
        moment_135_s.append(moment_135)
        moment_45_rels.append(moment_45_rel)
        moment_135_rels.append(moment_135_rel)

    indexes = [i for i in range(len(files))]
    columns = ['file', 'weight', 'normalized_weight', 'x_avg', 'y_avg',
               'x_avg_rel', 'y_avg_rel', 'moment_x', 'moment_y',
               'moment_x_rel', 'moment_y_rel', 'moment_45', 'moment_135',
               'moment_45_rel', 'moment_135_rel']
    data = list(zip(files, weights, normalized_weights, x_avgs, y_avgs, x_avg_rels, y_avg_rels,
            moment_x_s, moment_y_s, moment_x_rels, moment_y_rels, moment_45_s,
            moment_135_s, moment_45_rels, moment_135_rels))
    df = pd.DataFrame(data, indexes, columns)
    df.to_csv(f'{file_path}.csv')

def create_data(font_path, alphabet, path_name, test_count) :
    create_alphabet(font_path, alphabet, path_name)
    set_tests(font_path, alphabet, test_count)
    get_characterics(path_name)