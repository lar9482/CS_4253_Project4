import matplotlib.pyplot as plt
import os
import sys

def graph_DT_data(data, num_instances, domain):

    possibleColors =['r', 'b', 'g', 'm', 'c', 'k']
    currentColor = 0
    legendList = []

    plt.legend('', frameon=False)

    for info_gain_metric in data.keys():
        all_instances = []
        all_training_accuracies = []
        all_testing_accuracies = []
        for unit in data[info_gain_metric]:
            all_instances.append(unit[2])
            all_training_accuracies.append(unit[0])
            all_testing_accuracies.append(unit[1])

        plt.plot(all_instances, all_training_accuracies, color = possibleColors[currentColor], label = info_gain_metric.name + "_TN")
        currentColor += 1

        plt.plot(all_instances, all_testing_accuracies, possibleColors[currentColor], label = info_gain_metric.name + "_TT")
        currentColor += 1

        legendList.append(info_gain_metric.name + "_TN")
        legendList.append(info_gain_metric.name + "_TT")

    plt.legend(bbox_to_anchor=(1.15, 1.15), loc='upper right')
    plt.xlabel('Number of Instances')
    plt.ylabel('Accuracy')
    
    plt.xlim(0, 4600)
    plt.ylim(0, 1.05)

    filePath = os.path.join(sys.path[0], "Results", "DecisionTree", domain.__name__)
    plt.savefig(filePath)
    plt.clf()