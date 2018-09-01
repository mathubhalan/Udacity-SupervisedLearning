# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 19:40:21 2018

@author: Mathu_Gopalan
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

#import data
bmi_life_data = pd.read_csv("D:\\Mathu\\udacity\\Nanodgree-DS\\Code\\Linear_Reg\\bmi_and_life_expectancy.csv")
x_v = bmi_life_data["BMI"]
y_v = bmi_life_data["Life expectancy"]
x_n = x_v.values.reshape(163,1)
y_n = y_v.values.reshape(163,1)
model = LinearRegression()
model.fit(x_n,y_n)
model.predict(32.0)

bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])
aos_life_exp = bmi_life_model.predict(21.07931)
