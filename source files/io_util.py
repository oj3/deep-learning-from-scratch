import sys, os
import numpy as np

class NumpyIO:

    def __init__(self):
        self.enc = 'utf-8'

    # imports numpy array data from a text file
    def load(self, path):
        if (not os.path.exists(path)):
            return None

        dataSet = {}
        with open(path, 'r') as f:
            beginningLine = -1
            currentData = None
            dataName = None
            shape = None

            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i].strip()

                if (line == ''):
                    continue

                elif (line == 'BEGIN'):   # Beginning of a data set
                    beginningLine = i + 1

                elif (line == 'END'):     # End of a data set
                    if (dataName in dataSet):
                        dataSet[dataName + '\t@Line ' + str(beginningLine)] = self.__to_numpy_array(currentData, shape)
                    else:
                        dataSet[dataName] = self.__to_numpy_array(currentData, shape)

                    currentData = None
                    dataName = None
                    shape = None
                    beginningLine = -1

                elif (0 < beginningLine):
                    if (currentData == None):   # The data definition line: data name; shape
                        currentData = []
                        dataName, shape = self.__get_data_definition(line)
                    else:                       # A data
                        texts = line.split('\t')
                        for text in texts:
                            currentData.append(float(text))

            return dataSet

    # exports numpy array data to a text file: append the data if the file exists
    def save(self, path, name, param):
        self.__export(path, name, param, 'a')

    # exports numpy array data to a text file: the contents is fully replaced if the file exists
    def save_new(self, path, name, param):
        self.__export(path, name, param, 'w')

    # seq: The sequence of target number starting at 0
    def to_index(self, seq, shape):
        length = len(shape)

        q = seq
        r = 0
        coord = []
        for i in range(length):
            if (q == 0):
                coord.append(0)
                continue

            divisor = shape[length - 1 - i]
            r = q % divisor
            q = (int)(q / divisor)

            coord.append(r)

        coord.reverse()
        return tuple(coord)

    def __get_data_definition(self, line):
        definitions = line.split('\t')
        dataName = definitions[0]
        shape = []
        dimInfo = definitions[1].strip('(').strip(')').split(',')
        for axis in range(len(dimInfo)):
            size = dimInfo[axis].strip()
            if (not size == ''):
                shape.append(int(size))

        return (dataName, tuple(shape))

    def __to_numpy_array(self, data, shape):
        if (len(shape) == 1):
            return np.array(data)

        arraySize = 1
        for size in shape:
            arraySize *= size

        npArray = np.zeros(shape)
        for i in range(len(data)):
            if (i == arraySize):    # stop if the array is full
                break

            id = self.to_index(i, shape)
            npArray[id] = data[i]

        return npArray

    def __save(self, path, name, param, mode):
        shape = param.shape
        lineLength = (int)(param.size / shape[0]) if (1 < param.ndim) else 50

        isNewFile = os.path.exists(path)

        with open(path, mode) as f:
            if (isNewFile):
                f.write('\n')

            f.write('BEGIN' + '\n')
            f.write('\t' + name + '\t' + str(shape) + '\n')

            for i in range(param.size):
                index = self.to_index(i, shape)
                f.write('\t' + str(param[index]))
                if (0 < i and (i % lineLength) == 0):
                    f.write('\n')
                elif (i == param.size - 1):
                    f.write('\n')

            f.write('END' + '\n')

    # test methods - - - - - - - - - - - - - - - - - - -

"""
    # test method
    def get_data_definition(self, line):
        return self.__get_data_definition(line)

    # test method
    def __testof_to_index(self):
        y = np.array([[[1,2,3,4], [5,6,7,8], [9,10,11,12]], [[13,14,15,16], [17,18,19,20],[21,22,23,24]]])
        for i in range(y.size):
            seq = i + 1
            print(str(seq) + " --> " + str(self.to_index(seq, y.shape)))
"""