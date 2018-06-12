import numpy as np

def onehot(labels, depth):
    encoding = np.zeros([labels.size, depth])
    encoding[np.arange(labels.size), labels] = 1
    return encoding