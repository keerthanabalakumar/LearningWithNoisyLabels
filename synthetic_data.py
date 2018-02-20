import numpy as np
import pandas as pd


 
df = pd.read_csv('../Dataset/Diabetes_indians-Noisy-0.3-0.3.csv')


real_data =df[df.columns[-1]] 
noisy_data = [None]*len(real_data)

for i in range(len(real_data)):

    if (real_data[i]== 1):
    	flip_if_positive = np.random.choice(2,2,p=[0.3,0.7])
	if(flip_if_positive[0] == 0):
	    noisy_data[i] = 0
	else:
	    noisy_data[i] = real_data[i]
	
	
    elif (real_data[i]== 0):
    	flip_if_negative = np.random.choice(2,2,p=[0.3,0.7])
	if(flip_if_negative[0] == 0):
	    noisy_data[i] = 1

    	else:
	    noisy_data[i] = real_data[i]
for i in range(len(noisy_data)):
    print noisy_data[i]

