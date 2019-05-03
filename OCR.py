import numpy as np
import cv2
import pytesseract
from PIL import Image

#img = cv2.imread('result.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img3=cv2.imread('oly.png')
gray3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,180)]
testCells = [np.hsplit(row,36) for row in np.vsplit(gray3,1)]

# for row in np.vsplit(gray,180):
#  for col in np.hsplit(row, 100):
#   cv2.imshow("dd",col)
#   cv2.waitKey(0)

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)
y = np.array(testCells)
# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = y.reshape(-1,400).astype(np.float32) # Size = (2500,400)
# i=0;
# for row in test:
#  for col in row:   ## must remove . reshape .astype from test in order to appear pics
#   cv2.imwrite("C:\\Users\\Omar Anas\\PycharmProjects\\ImageProcessing\\testCases\\"+str(i)+".png",col)
#   i+=1

# Create labels for train and test data
k = np.arange(36)
train_labels = np.repeat(k,250)[:,np.newaxis]
# test_labels = train_labels.copy()
test_labels=np.repeat(k,36)
# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)
#
letters = dict({"0":"0", "1":"1","2":"2", "3":"3","4":"4", "5":"5","6":"6", "7":"7","8":"8", "9":"9",
    "10":"A", "11":"B","12":"C", "13":"D","14":"E", "15":"F","16":"G", "17":"H","18":"I", "19":"J"
                 ,"20":"K", "21":"L","22":"M", "23":"N","24":"O", "25":"P","26":"Q", "27":"R","28":"S", "29":"T"
                 ,"30":"U", "31":"V","32":"W", "33":"X","34":"Y", "35":"Z"})
numStr=""
for i in result:
 numStr+=str(i)

print (numStr)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
# matches = result==test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct*100.0/result.size
# print (accuracy)