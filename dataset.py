import pandas as pd
import numpy as np
import os.path, os
import utility

class Dataset:

    def __init__(self, train_path, test_path, masks_path):
        self.train_df, self.test_df = utility.build(train_path, test_path, masks_path)

    def draw(self, size):
        pass