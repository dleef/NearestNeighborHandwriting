"""
Authors: Peter Looi and Daniel Leef

"""
import sys
from ImageDataSearch import *
import numpy as np
from time import time
from math import atan
from math import pi
from random import random
from pyhull.voronoi import VoronoiTess

'''
Daniel Leef Contributions to this file:
All Gabriel Methods
Voronoi Method
Incremental Deletion and Growth Methods
image_to_class method in Nearest Neighbor class
'''

"""
Main function

Tests the nearest neighbor model

"""
def main():
    #you can change num_data if you would like
    #num_data is the number of data points in the training data
    #7000 is a good number because it gives pretty decent accuracy, while not taking too long
    num_training_data = 12000
    
    #you can also change num_validation_data
    #this is the number of data points in the validation data
    #cannot be over 1000, otherwise the 1000th data point onwards will be exactly the same as training data
    num_validation_data = 100 
    
    
    #get the training data
    #training data starts from image 1000 
    train = (get_data("emnist-letters-train.csv", range(1000, num_training_data+1000), "emnist-letters-mapping.txt"))
   
    #train = (get_average_data("emnist-letters-train.csv", range(1000, num_training_data+1000), "emnist-letters-mapping.txt"))

    #VORONOI and GABRIEL: (never produced any usable results)
    '''
    trainArray = (get_data_voronoi("emnist-letters-train.csv", range(1000, num_training_data+1000), "emnist-letters-mapping.txt"))
    voronoi(trainArray, train)

    gabriel_train = (get_data_gabriel("emnist-letters-train.csv", range(1000, num_training_data+1000), "emnist-letters-mapping.txt"))
    to_numpy(gabriel_train)
    origTrain = gabriel_train.copy()
    gabriel(gabriel_train, origTrain)
    '''

    #get the validation data
    #validation data is the first 50 images (first 1000 reserved)
    validation = (get_data("emnist-letters-train.csv", range(0, 0+num_validation_data), "emnist-letters-mapping.txt"))
    
    #convert the PIL images to numpy matricies. Numpy matricies allows for faster calculations
    to_numpy(train)
    to_numpy(validation)
    
    #apply a transformation to the image
    #scale_down with average with final_width 12 seems to help the most
    final_width = 12
    scale_down(train, final_width, "average")
    scale_down(validation, final_width, "average")

    #INCREMENTAL GROWTH
    #newTrain = incremental_growth(train)

    #INCREMENTAL DELETION
    #incremental_deletion(train)

    #create a NearestNeighbor object. This object has the functionality for the Nearest Neighbor model
    n = NearestNeighbor()
    
    #Test the model with the validation data (and feeding in the training data, of course, because this is nearest neighbor)
    
    acc = n.accuracy(train, validation, 
                k=1, #can change the k value (but I found that k = 1 is usually the best)
                verbose = True, #this tells the algorithm whether or not to print percentage complete
                distance_metric = euclidian_distance)#can change the distance metric
    
    #print the final accuracy value
    print("Accuracy = " + str(acc))


"""
Here are the various distance metrics

image1 and image2 are numpy matrixes
"""
def euclidian_distance(image1, image2):
    difs = image1 - image2
    return np.square(difs).sum()
def euclidian_distance_cube(image1, image2):
    difs = image1 - image2
    return np.absolute(np.power(difs, np.full(image1.shape, 3))).sum()
def euclidian_distance_for_angles(image1, image2):
    difs = image1 - image2
    #here we have the raw angle differences
    difs = np.absolute(difs)
    subtract = np.full(image1.shape, 2*pi)
    difs2 = difs - subtract
    difs2 = np.absolute(difs2)
    difs = np.minimum(difs, difs2)
    return np.square(difs).sum()
def abs(x):
    return x if x >= 0 else -x
def l1_distance(image1, image2):
    difs = image1 - image2
    return np.absolute(difs).sum()



"""
Instantiate a NearestNeighbor object to access the classifyer
"""
class NearestNeighbor:
    def __init__(self):
        pass
    """
    train is an array of training data
    test_array is an array of test data
    k = the k value in KNN
    distance_metric specifys how you would like to calculate distance (such as 
    euclidian distance or taxicab distance)
    
    returns a single float value between 0 and 1, representing the accuracy of the 
    nearest neigbor model on the test data
    """
    def accuracy(self, train, test_array, k=1, distance_metric=euclidian_distance, verbose=True):
        results = self.classify(train, test_array, k=k, distance_metric=distance_metric,verbose=verbose)
        total_images = len(test_array)
        total_correct = 0
        for i in range(len(test_array)):
            if get_label(test_array[i]) == results[i]:
                total_correct += 1
        return total_correct/total_images
    """
    train is an array of training data
    test_array is an array of test data
    k = the k value in KNN
    distance_metric specifys how you would like to calculate distance (such as 
    euclidian distance or taxicab distance)
    
    returns an array, where the ith index of the array is classification that 
    the nearest neigbor algorithm has given for the ith test data for
    """
    def classify(self, train, test_array, k=1, distance_metric=euclidian_distance, verbose=True):
        length = len(test_array)
        
        self.verify_train_test(train, test_array)
        results = []
        
        i = 0
        for test_input in test_array:
            #result = self.image_to_class(train, test_input, distance_metric=distance_metric)
            result = self.classify_one(train, test_input, k=k, distance_metric=distance_metric)
            results.append(result)
            if verbose: print(str(i/length* 100) + "%")
            i += 1
        return results
    """
    train is an array of training data
    test_array is one test data point
    k = the k value in KNN
    distance_metric specifys how you would like to calculate distance (such as 
    euclidian distance or taxicab distance)
    
    returns the classification that the nearest neigbor algorithm has given for 
    the test data 
    """
    def classify_one(self, train, test, k=1, distance_metric=euclidian_distance):
        top_k = [None]*k
        top_k_distances = [sys.float_info.max]*k
        for train_data in train:
            dist = distance_metric(get_image(train_data), get_image(test))
            
            #see if this training data can be added to the top k list
            for j in range(len(top_k_distances)-1, -1, -1):
                
                
                if dist >= top_k_distances[j]:#distances 0 to j are closer than dist
                    if j+1 < len(top_k_distances):#add it to the top_k_distances list at j+1 (not in 0 to j). If list is not long enough, forget it
                        top_k_distances.insert(j+1, dist)
                        del top_k_distances[-1]
                        
                        top_k.insert(j+1, train_data)
                        del top_k[-1]
                elif j == 0:
                    top_k_distances.insert(j, dist)
                    del top_k_distances[-1]
                    
                    top_k.insert(j, train_data)
                    del top_k[-1]
        
        #now, we will choose the classification that is the majority 
        frequency_dict = {}
        for data in top_k:
            if data == None: continue
            label = get_label(data)
            if label in frequency_dict:
                frequency_dict[label] = frequency_dict[label] + 1
            else:
                frequency_dict[label] = 1
        #find the majority
        classification = None
        frequency = -9999999999999
        for label in frequency_dict:
            freq = frequency_dict[label]
            if freq > frequency:
                frequency = freq
                classification = label
                
        return classification
        
    def image_to_class(self, averages, test, distance_metric=euclidian_distance):
        lowest_dist = 100000000
        label = ""
        for data in averages:
            dist = distance_metric(get_image(data), get_image(test))
            if (dist < lowest_dist):
                lowest_dist = dist
                label = data["label"]

        return label 


    def verify_train_test(self, train, test_array):
        height = -1
        width = -1
        for i in range(len(train)):
            data = train[i]
            if width == -1 and height == -1:
                height = get_height(data)
                width = get_width(data)
            elif width == get_width(data) and height == get_height(data):
                pass #good
            else:
                raise Exception("Training data point number " + str(i) + " has dimensions of " + str(get_width(data)) + ", " + str(get_height(data)) + " while the first training data point has dimensions of " + str(width) + ", " + str(height))
        for i in range(len(test_array)):
            data = test_array[i]
            if width == get_width(data) and height == get_height(data):
                pass #good
            else:
                raise Exception("Test data point number " + str(i) + " has dimensions of " + str(get_width(data)) + ", " + str(get_height(data)) + " while the first training data point has dimensions of " + str(width) + ", " + str(height))
    
  
"""
get_label, get_height, get_width may change, depending if the interface of
the data point changes
"""
def get_label(data):
    return data["label"]
def get_image(data):
    return data["image"]
def get_height(data):
    height, width = data["image"].shape
    return height
def get_width(data):
    height, width = data["image"].shape
    return width
def to_numpy(data_array):
    for data in data_array:
        img = data["image"]
        width, height = img.size
        matrix = []
        for y in range(height):
            row = []
            for x in range(width):
                row.append(img.getpixel((x, y)))
            matrix.append(row)
        matrix = np.matrix(matrix)
        data["image"] = matrix
    return data_array
            
def scale_down(dataset, final_width : "has to be square, sorry, will fix later", method):
    smaller_size = final_width
    for data in dataset:
        image = get_image(data)
        #p(image)
        width = get_width(data)
        height = get_height(data)
        block_width = get_width(data)//smaller_size
        #print(block_width)
        m = np.zeros((smaller_size, smaller_size))
        for block_row in range(smaller_size):
            for block_column in range(smaller_size):
                if method == "max":
                    maximum_value = 0
                    for x in range(block_column*block_width, block_column*block_width+block_width):
                        for y in range(block_row*block_width, block_row*block_width+block_width):
                            if image.item((y,x)) > maximum_value:
                                maximum_value = image.item((y,x))
                    m.itemset((block_row, block_column), maximum_value)
                elif method == "average":
                    average_value= 0
                    for x in range(block_column*block_width, block_column*block_width+block_width):
                        for y in range(block_row*block_width, block_row*block_width+block_width):
                            average_value += image.item((y,x))
                    average_value /= block_width**2
                    m.itemset((block_row, block_column), average_value)
                elif method == "min":
                    min_value = 999999999999999999999
                    for x in range(block_column*block_width, block_column*block_width+block_width):
                        for y in range(block_row*block_width, block_row*block_width+block_width):
                            if image.item((y,x)) < min_value:
                                min_value = image.item((y,x))
                    m.itemset((block_row, block_column), min_value)
                else:
                    raise Exception("method must be 'average' or 'max' or 'min'")
        data["image"] = m
        #p(m)
        #quit()
    return dataset
    
"""
doesn't help
"""
def add_noise(dataset):
    for data in dataset:
        image = get_image(data)
        width = get_width(data)
        height = get_height(data)
        for x in range(width):
            for y in range(height):
                if image.item((y,x)) < 30:
                    image.itemset((y,x), random() * 30)
                
    return dataset
    
    
def filter(dataset):
    for data in dataset:
        image = get_image(data)
        width = get_width(data)
        height = get_height(data)


        for x in range(width):
            for y in range(height):
                if image.item((y, x)) > 50:
                    image.itemset((y, x), 255)
                else:
                    image.itemset((y, x), 0)
    return dataset
def p(matrix, divide_by_pi=False):
    height, width = matrix.shape
    for y in range(height):
        for x in range(width):
            if divide_by_pi:
                chars = len(str(round(matrix.item((y,x))/pi, 3)))
                spaces = 8-chars
                print(str(round(matrix.item((y,x))/pi, 3)) + " "*spaces, end="")
                print("p, ", end="")
            else:
                chars = len(str(round(matrix.item((y,x)), 3)))
                spaces = 5-chars
                print(str(round(matrix.item((y,x)), 3))+ " "*spaces, end="")
                print(", ", end="")
        print()
    print("\n")

#Goal of this method was to initially just store all the points with similar vertices in
#the Voronoi diagram to one another, and eventually I was going to use this information to
#find out which points were surrounded by regions with points with the same label.
#These points would then get deleted from the training data set as they are unnecessary
'''
def voronoi(train, trainWithLabels):
    vor = VoronoiTess(train)
    points = vor.points
    vertices = vor.vertices
    ridges = vor.ridges.items()
    i = 0
    similarCounts = []
    for _ in range(len(train)):
        similarCounts.append(0)
    while (i < len(train)):
        first = train[i]
        if (i != len(train)-1):
            j = i + 1
            while (j < len(train)):
                second = train[j]
                vor2 = VoronoiTess([first, second])
                vertices2 = vor2.vertices
                for v in vertices2:
                    if (v in vertices):
                        if (trainWithLabels[i]["label"] == trainWithLabels[i+1]["label"]):
                            similarCounts[i] += 1
                            similarCounts[j] += 1
                j += 1
        i += 1
    print(similarCounts)
'''

def gabriel(train, origTrain):
    neighbors = gabriel_neighbors(train, origTrain)
    print(neighbors)
    to_be_deleted = []
    tracker = 0
    for z in range(len(train)):
        print("Percentage Done (gabriel): ", (tracker / len(train)))
        if (len(neighbors[z]) > 0):
            ref_label = origTrain[z]["label"]
            count = 0
            for n in neighbors:
                if (origTrain[n]["label"] == ref_label):
                    count += 1
            if (count > (0.5*len(neighbors))):
                to_be_deleted.append(z)
        tracker += 1

    print(to_be_deleted)


def gabriel_points(train):
    points = []
    i = 0
    while (i < len(train)-1):
        print("Percentage Done (Points): ", (i/(len(train)-1)))
        j = 0
        while (j < len(train)-1):
            if (j == i and i == len(train)-2):
                break
            elif (j == i and i != len(train)-2):
                j += 1
            ti = get_image(train[i])
            tj = get_image(train[j])
            dist = euclidian_distance(ti, tj)
            point1 = ti + (dist/2.0)
            point2 = ti - (dist/2.0)
            point3 = tj + (dist/2.0)
            point4 = tj - (dist/2.0)
            points.append({"i" : i, "j" : j, "point1" : point1, "point2" : point2, "point3" : point3, "point4" : point4})
            j+=1
        i += 1
    return points

def gabriel_neighbors(train, origTrain):
    gabrielNeighbors = {}
    for i in range(len(train)):
        gabrielNeighbors[i] = []
    points = gabriel_points(train)
    tracker = 0
    for point in points:
        print("Percentage Done (Neighbors): ", (tracker/len(points)) , "%")
        i = point["i"]
        j = point["j"]
        point1 = point["point1"]
        point2 = point["point2"]
        point3 = point["point3"]
        point4 = point["point4"]
        newTrain = train.copy()
        del newTrain[i]
        del newTrain[j]
        for t in newTrain:
            tt = get_image(t)
            if (((tt <= point1) ^ (tt >= point2)).any() or ((tt <= point3) ^ (tt >= point4)).any()):
                print("whoa")
                gab = False
                break
            else:
                gab = True
        if (gab):
            gabrielNeighbors[i].append(j)

        tracker += 1
    return gabrielNeighbors


def incremental_deletion(train):
    n = NearestNeighbor()
    output = []
    index = 0
    for train_data in train:
        nTrain = train.copy()
        print("Length of training: ", len(train))
        del nTrain[index]
        #Changing K value in function below leads to varying results
        classification = n.classify_one(nTrain, train_data, k = 9, distance_metric = euclidian_distance)
        if (classification == train_data["label"]):
            del train[index]
        else:
            index += 1
            print("index: ", index)
    print(len(train))
    

def incremental_growth(train):
    n = NearestNeighbor()
    output = []
    for train_data in train:
        #Changing K value in function below leads to varying results
        classification = n.classify_one(output, train_data, k = 3, distance_metric = euclidian_distance)
        if (classification != train_data["label"]):
            output.append(train_data)
        print(len(output))
    print(len(output))
    return output


if __name__ == "__main__": main()


"""
def image_gradient(dataset):
    for data in dataset:
        image = get_image(data)
        width = get_width(data)
        height = get_height(data)
        if width != 25 or height != 25:
            raise Exception()
            
        smaller_width = 5
        block_width = 5
        #p(image)
        m = np.zeros((smaller_width, smaller_width))
        for block_row in range(smaller_width):
            for block_column in range(smaller_width):
                center_x, center_y = block_column*block_width+((block_width-1)/2), block_row*block_width+((block_width-1)/2)
                average_x = 0
                average_y = 0
                total_magnitude_in_block = 0
                for x in range(block_column*block_width, block_column*block_width+block_width):
                    for y in range(block_row*block_width, block_row*block_width+block_width):
                        magnitude = image.item((y,x))
                        average_x += magnitude * x
                        average_y += magnitude * y
                        total_magnitude_in_block += magnitude
                if total_magnitude_in_block == 0:
                    average_x = center_x
                    average_y = center_y
                else:
                    average_x /= total_magnitude_in_block
                    average_y /= total_magnitude_in_block
                
                dy = -(average_y - center_y) #negative because in the image y gets larger as we go down, but in unit circle y gets larger as we go up
                dx = average_x - center_x
                #print("Block", block_row, block_column, "ax=", average_x, "ay=", average_y, "cx", center_x, "cy", center_y)
                #print("Block", block_row, block_column, "dy", dy, "dx", dx)
                if dy == 0 and dx == 0:
                    angle = 0
                elif dx == 0:
                    if dy > 0:
                        angle = pi/2#90deg
                    else:#dy < 0
                        angle = -pi/2#-90deg
                else:
                    angle = atan(dy/dx)
                if dx < 0:
                    angle += pi#+180deg
                
                m.itemset((block_row, block_column), angle)
        #p(m, True)
        #quit()
        data["image"] = m
    return dataset


def image_gradient(dataset):
    for data in dataset:
        image = get_image(data)
        width = get_width(data)
        height = get_height(data)
        if width != 25 or height != 25:
            raise Exception()
            
        smaller_width = 5
        block_width = 5
        #p(image)
        m = np.zeros((smaller_width, smaller_width))
        for block_row in range(smaller_width):
            for block_column in range(smaller_width):
                center_x, center_y = block_column*block_width+((block_width-1)/2), block_row*block_width+((block_width-1)/2)
                average_x = 0
                average_y = 0
                total_magnitude_in_block = 0
                for x in range(block_column*block_width, block_column*block_width+block_width):
                    for y in range(block_row*block_width, block_row*block_width+block_width):
                        magnitude = image.item((y,x))
                        average_x += magnitude * x
                        average_y += magnitude * y
                        total_magnitude_in_block += magnitude
                if total_magnitude_in_block == 0:
                    average_x = center_x
                    average_y = center_y
                else:
                    average_x /= total_magnitude_in_block
                    average_y /= total_magnitude_in_block
                
                dy = -(average_y - center_y) #negative because in the image y gets larger as we go down, but in unit circle y gets larger as we go up
                dx = average_x - center_x
                #print("Block", block_row, block_column, "ax=", average_x, "ay=", average_y, "cx", center_x, "cy", center_y)
                #print("Block", block_row, block_column, "dy", dy, "dx", dx)
                if dy == 0 and dx == 0:
                    angle = 0
                elif dx == 0:
                    if dy > 0:
                        angle = pi/2#90deg
                    else:#dy < 0
                        angle = -pi/2#-90deg
                else:
                    angle = atan(dy/dx)
                if dx < 0:
                    angle += pi#+180deg
                
                m.itemset((block_row, block_column), angle)
        #p(m, True)
        #quit()
        data["image"] = m
    return dataset
def main2():
    data = [{"label" : "", "image" : np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    
    
    ])}]
    
    image_gradient(data)
    def distance_to_nearest_pixel_in_other(image1, image2, threshold = 50, search_radius=4):
    height, width = image1.shape
    
    
    
    distance = 0
    for x in range(width):
        for y in range(height):
            if image1.item((x,y)) > threshold:
                found_nearby_darkened_pixel = False
                for r in range(search_radius):
                    for compare_x, compare_y in loop(x, y, r):
                        if compare_x >= 0 and compare_x < width and compare_y >= 0 and compare_y < height:
                            if image2.item((compare_x, compare_y)) > threshold:
                                #pixel x,y in image1 is darkened, and there is a nearby
                                #pixel at compare_x, compare_y in image2 that is darkened,
                                #and it is at r radius away
                                distance += ( (compare_y-y)**2 + (compare_x-x)**2 ) ** 4 #HEAVILY punish far away pixels
                                found_nearby_darkened_pixel = True
                                break
                    if found_nearby_darkened_pixel:
                        break
                if not found_nearby_darkened_pixel:
                    distance += 2*(2*(search_radius+1) ** 2)**6#extra penalty for being far
    
    
    temp = image1
    image1 = image2
    image2 = temp
    #now, do the same thing but for image2 on image1
    for x in range(width):
        for y in range(height):
            if image1.item((x,y)) > threshold:
                found_nearby_darkened_pixel = False
                for r in range(search_radius):
                    for compare_x, compare_y in loop(x, y, r):
                        if compare_x >= 0 and compare_x < width and compare_y >= 0 and compare_y < height:
                            if image2.item((compare_x, compare_y)) > threshold:
                                #pixel x,y in image1 is darkened, and there is a nearby
                                #pixel at compare_x, compare_y in image2 that is darkened,
                                #and it is at r radius away
                                distance += ( (compare_y-y)**2 + (compare_x-x)**2 ) ** 4 #HEAVILY punish far away pixels
                                found_nearby_darkened_pixel = True
                                break
                    if found_nearby_darkened_pixel:
                        break
                if not found_nearby_darkened_pixel:
                    distance += 2*(2*(search_radius+1) ** 2)**6#extra penalty for being far
    return distance
def main2():
    image1 = np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]])
    image2 = np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,1,0,0,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]])
    image3 = np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]])
    print(distance_to_nearest_pixel_in_other(image1, image2, threshold=0))
def loop(center_x, center_y, radius):
    min_x = center_x - radius
    min_y = center_y - radius
    max_x = center_x + radius
    max_y = center_y + radius
    
    yield min_x, center_y
    yield max_x, center_y
    yield center_x, max_y
    yield center_x, min_y
    
    left_up = 1
    left_down = 1
    right_up = 1
    right_down = 1
    up_left = 1
    up_right = 1
    down_left = 1
    down_right = 1
    
    while True:
        if left_up > radius: break
        yield min_x, center_y+left_up
        left_up += 1
        yield min_x, center_y-left_down
        left_down += 1
        yield max_x, center_y+right_up
        right_up += 1
        yield max_x, center_y-right_down
        right_down += 1
        
        if up_right < radius:
            yield center_x+up_right, max_y
            up_right += 1
            yield center_x-up_left, max_y
            up_left += 1
            yield center_x+down_right, min_y
            down_right += 1
            yield center_x-down_left, min_y
            down_left += 1
        
"""
