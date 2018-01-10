import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
import scipy.optimize as opt

labelencoder=LabelEncoder()

f = pd.read_csv('mushrooms.csv')
xy = DataFrame(f)

print(xy.dtypes)
