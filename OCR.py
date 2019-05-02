import numpy as np
import cv2
import pytesseract
from PIL import Image

#
img = cv2.imread('result.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,180)]

# for row in np.vsplit(gray,180):
#  for col in np.hsplit(row, 100):
#   cv2.imshow("dd",col)
#   cv2.waitKey(0)

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# i=0;
# for row in test:
#  for col in row:   ## must remove . reshape .astype from test in order to appear pics
#   cv2.imwrite("C:\\Users\\Omar Anas\\PycharmProjects\\ImageProcessing\\testCases\\"+str(i)+".png",col)
#   i+=1

# Create labels for train and test data
k = np.arange(36)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)
#
for i in result:
 print(i)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print (accuracy)