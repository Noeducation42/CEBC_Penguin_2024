import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os



print("Répertoire de travail actuel :", os.getcwd())


#%% 

name = str('D:\PenguinsData\LittlePenguin\2020\Guard')


df = pd.read_csv('D:\PenguinsData\LittlePenguin\2020\Guard\20I3093MP23_S1.csv')
pressure_values = df['Pressure'].dropna()
pressure_list = pressure_values.tolist()
print(pressure_list)