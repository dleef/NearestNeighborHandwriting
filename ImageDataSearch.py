"""
DATASET: Use emnist-balanced
Authors: Daniel Leef and Peter Looi
"""


from __future__ import print_function
import os, sys
from PIL import Image
import scipy.io
import re
import sys
import numpy as np
from io import StringIO

#Peter Method
def get_mapping(mapping_file):
    mapping = {}
    f = open(mapping_file)
    for line in f:
        line =  line.strip().split(" ")
        mapping[int(line[0])] = chr(int(line[1]))
  
    return mapping

#Daniel Method
'''
def get_data_voronoi(file_name, iterator, mapping_file):
    pixelArray = []
    if mapping_file == None:
        class NoMapping:
            def __getitem__(self, key):
                return key
        mapping = NoMapping()
    else:
        mapping = get_mapping(mapping_file)
    
    data = []
    f = open(file_name, 'r')
    p = re.compile(',')
    lines = f.readlines()
    for i in iterator:
        l = lines[i]
        example = [int(x) for x in p.split(l.strip())]
        pixels = example[1:]
        letter = example[0]
        size = 8,8
        img = Image.new('L', (28, 28))
        img.putdata(pixels)
        img.thumbnail(size, Image.ANTIALIAS)
        img = img.rotate(-90)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        pixelArray.append(img.getdata())
    return pixelArray
'''
#Daniel Method
'''
def get_data_gabriel(file_name, iterator, mapping_file):
    if mapping_file == None:
        class NoMapping:
            def __getitem__(self, key):
                return key
        mapping = NoMapping()
    else:
        mapping = get_mapping(mapping_file)
    
    data = []
    f = open(file_name, 'r')
    p = re.compile(',')
    lines = f.readlines()
    for i in iterator:
        l = lines[i]
        example = [int(x) for x in p.split(l.strip())]
        pixels = example[1:]
        letter = example[0]
        size = 12,12
        img = Image.new('L', (28, 28))
        img.putdata(pixels)
        img.thumbnail(size, Image.ANTIALIAS)
        img = img.rotate(-90)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        data.append({"label" : str(mapping[letter]), "image" : img})
    return data
'''
#Daniel Method
def get_average_data(file_name, iterator, mapping_file):
    averages = {}
    output = []
    letter_numbs = find_numb_each_letter(file_name, iterator, mapping_file)
    if mapping_file == None:
        class NoMapping:
            def __getitem__(self, key):
                return key
        mapping = NoMapping()
    else:
        mapping = get_mapping(mapping_file)
    
    f = open(file_name, 'r')
    p = re.compile(',')
    lines = f.readlines()
    N = len(lines)
    for i in iterator:
        l = lines[i]
        example = [int(x) for x in p.split(l.strip())]
        pixels = example[1:]
        letter = example[0]
        size = 25,25
        img = Image.new('L', (28, 28))
        img.putdata(pixels)
        img.thumbnail(size, Image.ANTIALIAS)
        img = img.rotate(-90)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if (str(mapping[letter]) in averages):
            averages[str(mapping[letter])].append(img)
        else:
            averages[str(mapping[letter])] = []
            averages[str(mapping[letter])].append(img)
    for key in averages:
        images = np.array([np.array(pic) for pic in averages[key]])
        average_image = np.zeros((25, 25), np.float)
        for i in images:
            average_image += i / letter_numbs[key]
        arr = np.array(np.round(average_image), dtype = np.uint8)
        #arr = np.array(np.mean(images, axis = 0), dtype = np.uint8)
        output.append({"label" : key, "image" : Image.fromarray(arr)})
    return output

#Daniel Method
def find_numb_each_letter(file_name, iterator, mapping_file):
    letter_numbs = {}
    if mapping_file == None:
        class NoMapping:
            def __getitem__(self, key):
                return key
        mapping = NoMapping()
    else:
        mapping = get_mapping(mapping_file)
    
    f = open(file_name, 'r')
    p = re.compile(',')
    lines = f.readlines()
    for i in iterator:
        l = lines[i]
        example = [int(x) for x in p.split(l.strip())]
        letter = example[0]
        string_letter = str(mapping[letter])
        if (string_letter in letter_numbs):
            letter_numbs[string_letter] += 1
        else:
            letter_numbs[string_letter] = 1

    return letter_numbs

#Daniel and Peter Method
def get_data(file_name, iterator, mapping_file):
    if mapping_file == None:
        class NoMapping:
            def __getitem__(self, key):
                return key
        mapping = NoMapping()
    else:
        mapping = get_mapping(mapping_file)
    
    data = []
    f = open(file_name, 'r')
    p = re.compile(',')
    lines = f.readlines()
    for i in iterator:
        l = lines[i]
        #Daniel Code Begin
        example = [int(x) for x in p.split(l.strip())]
        #the pixel values are 784 columns of the 785 in each row of the CSV file            
        pixels = example[1:]
        #the letter classification is the first column of the 785, it's a number 1-26 
        #correlating to a letter in the alphabet
        letter = example[0]
        size = 25,25
        #the 'L' is used to define the new image as a greyscale one
        img = Image.new('L', (28, 28))
        img.putdata(pixels)
        #this line converts the image to a 25 by 25 one from a 28 by 28
        img.thumbnail(size, Image.ANTIALIAS)
        #Daniel Code End
        img = img.rotate(-90)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        data.append({"label" : str(mapping[letter]), "image" : img})
    return data

#All Peter Code:
def show(data_array, index):
    data_array[index]["image"].show()
def show2(file_name, index):
    show(get_data(file_name, [index], None), 0)
def getTrainArray(pixels):
    return pixels

def main():
    data = get_data("emnist-letters-train.csv", range(0,100), "emnist-letters-mapping.txt")

if __name__ == "__main__":
  main()
