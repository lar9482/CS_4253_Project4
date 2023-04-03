import os
import sys
import numpy as np

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

    X = np.empty((instances, 57))
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