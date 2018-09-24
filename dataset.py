import pandas as pd
import numpy as np
import os.path, os
import utility

class Dataset:

    def __init__(self, train_path, test_path, masks_path):
        self.train_df, self.test_df = utility.build(train_path, test_path, masks_path)

    def draw(self, size, training, random_state=None):
        return self._sample(size, training, random_state)
    
    def _sample(self, size, training, random_state):
        s = None
        temp = None
        if training:
            temp = self.train_df.drop(pd.DataFrame())
        else:
            temp = self.test_df.drop(pd.DataFrame())

        while len(temp) > 0:
            if len(temp) < size:
                s = temp.sample(len(temp), random_state=random_state)
                temp = temp.drop(s.index)

                s = utility.convert(s)

                yield utility.reform(s)
            else:
                s = temp.sample(size, random_state=random_state)
                temp = temp.drop(s.index)

                s = utility.convert(s)

                yield utility.reform(s)