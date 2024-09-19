# -*- coding: utf-8 -*-
"""
Create distribution of women over relavant variables
"""
import pandas
import numpy as np


#Create first the dataset to be filled with frequencies
#tratment 0 or 1
#age first child 0 to 9 (10 possibilities)
#first year 1981 to 2007 (27)
#last year 1981 to 2007 (27)
#frequency

data=np.zeros((2*21*27*27,5))

i=0
for t in range(0,2):
    for a in range(0,21):
        for yi in range(1981,2008):
            for yf in range(1981,2008):
            
                
                data[i,0]=t
                data[i,1]=a
                data[i,2]=yi
                data[i,3]=yf
               
                
                i+=1
                
#Now fill in the dataset
i=0
for t in range(0,2):
    for a in range(0,21):
        
        try:
            current_data = pandas.read_excel('treated_'+str(t)+'_age_'+str(a)+'.xlsx')
        except:
            print('No dataset')
            
        for yi in range(1981,2008):
            for yf in range(1981,2008):
                
                
                try:
                    row=np.array(current_data)[:,0]==yi 
                    data[i,4] = np.array(current_data[yf][row])[0]
                except:
                    print('Empty entry')
                
                i+=1
                print(i)
                
#Drop empy entries
data[data[:,-1]==0]

#Save to pandas
df=pandas.DataFrame(data=data,columns=['treated','age','yi','yf','freq'])

#Drop entries with zero frequency
data_panda = df.drop(df[(df['freq']==0)].index)

#Save
data_panda.to_excel("C:/Users/32489/Dropbox/occupation/model/pfabio/frequencies.xlsx")  