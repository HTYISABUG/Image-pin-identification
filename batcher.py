import os
import re
from collections import namedtuple

import pandas as pd
import cv2
import numpy as np

def batch_generator(data, batch_size):

    while True:
        samples = np.random.choice(len(data.images), size=batch_size, replace=False)

        yield namedtuple('Dataset', ['images', 'labels'])(**{'images': data.images[samples], 'labels': data.labels[samples]})
