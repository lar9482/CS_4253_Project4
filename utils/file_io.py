import os
import sys
import numpy as np
import pandas as pd
from utils.shuffle import shuffle

def load_artificial_dataset(instances = 1000):
    filePath = os.path.join(sys.path[0], "Datasets", "Artificial", "labeled-examples.txt")

    read_items = 0

    global_X = np.empty((1000, 2))
    global_Y = np.empty((1000, 1))

    X = np.empty((instances, 2))
    Y = np.empty((instances, 1))

    with open(filePath, "r") as f:
        while True:
            content = f.readline().split(' ')
            if content[0] == '':
                break
            
            global_X[read_items, :] = [float(content[1]), float(content[2])]
            global_Y[read_items, :] = [float(content[0])]

            read_items += 1
        f.close()
    
    (global_X, global_Y) = shuffle(global_X, global_Y)

    indices = np.random.randint(instances, size=(instances))
    X = np.array([global_X[indice, :] for indice in indices])
    Y = np.array([global_Y[indice, :] for indice in indices])
    
    return (X, Y)


def load_EMG_data(instances = 5000, folder = "sub2"):
    parent_folder = "EMG Physical Action Data Set"
    classifications = ["Elbowing.txt", "Frontkicking.txt", "Hamering.txt", "Headering.txt", "Kneeing.txt",
                       "Pulling.txt", "Punching.txt", "Pushing.txt", "Sidekicking.txt", "Slapping.txt",
                       "Bowing.txt", "Clapping.txt", "Handshaking.txt", "Hugging.txt", "Jumping.txt",
                       "Running.txt", "Seating.txt", "Standing.txt", "Walking.txt", "Waving.txt"]

    subsection = int(instances/len(classifications))

    X = np.empty((instances, 8))
    Y = np.empty((instances, 1))
    curr_i = 0
    for class_name in range(0, len(classifications)):
        
        file_path = os.path.join(sys.path[0], "Datasets", parent_folder, folder, classifications[class_name])
        items_read = 0

        with open(file_path, "r") as f:
            while True:
                #Getting a row from the file
                content = f.readline().split('\t')    

                #If there are no more rows left, or the maximum subsection has been read, move on to the next file  
                if content[0] == '' or items_read >= subsection:
                    break      
                
                #Convert content list into a list of floats
                content = [float(attri) for attri in content]

                X[curr_i, :] = content
                Y[curr_i, :] = [class_name]
                                        
                items_read += 1
                curr_i += 1


            f.close()
        
    return (X, Y)

def load_optdigits_data(instances = 5000):
    file_path = os.path.join(sys.path[0], "Datasets", "Optical MNIST", "optdigits_whole.txt")
    items_read = 0

    X = np.empty((instances, 64))
    Y = np.empty((instances, 1))

    with open(file_path, "r") as f:
        while True:
            #Getting a row from the file
            content = f.readline().split(',')   

            #If there are no more rows left, or the maximum subsection has been read, move on to the next file  
            if content[0] == '' or items_read >= instances:
                break 

            #Convert content list into a list of floats
            content = [float(attri) for attri in content]

            #Remove the last element from content, which is the classification
            class_name = content.pop()

            X[items_read, :] = content
            Y[items_read, :] = [class_name]
            
            items_read += 1
        
        f.close()

    return (X, Y)

def load_spambase_data(instances = 4500):
    file_path = os.path.join(sys.path[0], "Datasets", "Spambase", "spambase_data.data")
    items_read = 0

    global_X = np.empty((4601, 57))
    global_Y = np.empty((4601, 1))
    X = np.empty((instances, 57))
    Y = np.empty((instances, 1))
    with open(file_path, "r") as f:
        while True:
            #Getting a row from the file
            content = f.readline().split(',')   
            #If there are no more rows left, or the maximum subsection has been read, move on to the next file  
            if content[0] == '':
                break 
            
            content = [float(attri) for attri in content]

            #Remove the last element from content, which is the classification
            class_name = content.pop()

            global_X[items_read, :] = content
            global_Y[items_read, :] = [class_name]
            
            items_read += 1
        
        (global_X, global_Y) = shuffle(global_X, global_Y)

        indices = np.random.randint(instances, size=(instances))
        X = np.array([global_X[indice, :] for indice in indices])
        Y = np.array([global_Y[indice, :] for indice in indices])
        
        f.close()
    
    return (X, Y)


def save_NN_data(data, domain_name):
    df = pd.DataFrame.from_dict(data)
    filePath = os.path.join(sys.path[0], "Results", "NN", domain_name + '.xlsx')
    df.to_excel(filePath,sheet_name='data')
    
