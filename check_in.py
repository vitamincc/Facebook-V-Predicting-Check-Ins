import numpy as np
import pandas as pd
import datetime
import csv
from collections import defaultdict, OrderedDict
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


numX = 2
numY = 2



#%% processing in a single grid
def grid_place_class(trainX, trainy, testX):
    
    #prediction place id for testData
    le = LabelEncoder()
    trainY = le.fit_transform(trainy.values)

    clf = SGDClassifier(loss='log')
    clf.fit(trainX.values, trainY)
    pred_y = clf.predict_proba(testX.values)
    pred_labels = le.inverse_transform(np.argsort(pred_y, axis=1)[:,::-1][:,0:3])
    row_index = testX.index 
    
    
    return (row_index, pred_labels) 

#%%
orgData = defaultdict(list)
with open('sample2.csv', 'r') as file:
    csv_reader = csv.reader(file)
    csv_reader.next()
    delta = 1E-8
    for row in csv_reader:
        orgData[row[0]] = [float(row[1]),float(row[2]), float(row[1])/(float(row[2])+delta), 
                           float(row[3]),float(row[4]),row[5]]
          
        date = datetime.datetime.fromtimestamp(float(row[4]))
        orgData[row[0]].append(date.year)
        orgData[row[0]].append(date.month)
        orgData[row[0]].append(date.day)
        orgData[row[0]].append(date.hour)
        orgData[row[0]].append(date.minute)
        orgData[row[0]].append(date.second)
        
        
compData = pd.DataFrame.from_dict(orgData, orient = "index")
compData.columns = ['x', 'y', 'xDy', 'accuracy','time','placeId','year', 'month','day','hour','min','second']

compData.to_csv('new_sample2.csv')
#%%

#max value of time
maxTime = np.max(compData['time'])
minTime = np.min(compData['time'])

maxAcc = np.max(compData['accuracy'])
minAcc = np.min(compData['accuracy'])

#boxplot, check outlier
boxData = compData[['accuracy','time']]
boxData.plot.box()


#calculate corresponding grid for each (x,y)
x1 = (compData['x']//(10/numX)).astype(int) + 1
x2 = (compData['x']//(10/numX)).astype(int)
df = np.where(compData['x']/(10/numX) >0, x1, x2)
compData['grid_x'] = df

y1 = (compData['y']//(10/numY)).astype(int) + 1
y2 = (compData['y']//(10/numY)).astype(int)
df = np.where(compData['y']/(10/numY) >0, y1, y2)
compData['grid_y'] = df



#%%Normalization
col = ['x', 'y', 'xDy', 'accuracy', 'year','month','day','hour','min','second']
for c in col:
    compData[c] = (compData[c].values- compData[c].mean())/compData[c].std()
    

#split traning data and testing data
train, test = train_test_split(compData, test_size=0.2, random_state =0)



threshold = 50

myResult = dict()
testY = dict()
for i in range(numX):
    for j in range(numY):
    	#to calculate which grid train data belongs to,
    	#and get all train data in this grid togehter
        gridTrain = train.loc[(train['grid_y'] == i)
                        & (train['grid_y'] == j)]
        
        #there must be at least 500 events in this grid, we do the analysis
        place_counts = gridTrain.placeId.value_counts()
        N_events = place_counts[gridTrain.placeId.values] >= threshold
        
        
        #pick up gridTrainX and gridTrainy
        gridTrainX = train[col]
        gridTrainy = train[['placeId']]
    
        #pick up gridTest, gridTestX and gridTesty
        #if a grid number been decided, then all the 
        #test data in this grid would put together
        gridTest =  test.loc[(test['grid_y'] == i)
                        & (test['grid_y'] == j)]
        gridTestX = test[col]       #get features for test data
        gridTesty = test[['placeId']]    #get actual placeId of test data

        #get rowIDs, placeIDs,matching EventId and placeId as kaggle asked 
        rowId, placeId = grid_place_class(gridTrainX, gridTrainy, gridTestX)
        rowId = rowId.values
    
        grid_result = defaultdict(list)
        grid_y = dict()
        #got the placeId
        str_labels = np.apply_along_axis(lambda x: ' '.join(x.astype(str)), 1, placeId)
        
        
        #event and placeId matched
        for i in range(len(temp)):
            grid_result[temp[i]] = str_labels[i] #slecet the correspoind placeId
            
            #get acutal placeId of test data for next step of confusion matrix
            grid_y[temp[i]] = gridTesty.values.tolist()[0] 
            
        myResult.update(grid_result)
        testY.update(grid_y)
       
        
#%%calcluate confusion matrix


pred = {}

actual = {}
for rowId, placeId in myResult.items():
    for i in range(3):
        x = placeId.split()
        if testY[rowId][0] == x[i]:
        	pred[rowId] = x[i]
        	actual[rowId] = x[i]
        else:
        	pred[rowId] = x[0]
        	actual[rowId] = testY[rowId][0]

cf = confusion_matrix(actual,pred)



            
        
    

        

        
        











