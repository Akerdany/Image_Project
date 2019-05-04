import numpy as np
import cv2
import pytesseract
from PIL import Image

letters = dict({"[0.]":"0", "[1.]":"1","[2.]":"2", "[3.]":"3","[4.]":"4", "[5.]":"5","[6.]":"6", "[7.]":"7","[8.]":"8", "[9.]":"9",
    "[10.]":"A", "[11.]":"B","[12.]":"C", "[13.]":"D","[14.]":"E", "[15.]":"F","[16.]":"G", "[17.]":"H","[18.]":"I", "[19.]":"J"
                 ,"[20.]":"K", "[21.]":"L","[22.]":"M", "[23.]":"N","[24.]":"O", "[25.]":"P","[26.]":"Q", "[27.]":"R","[28.]":"S", "[29.]":"T"
                 ,"[30.]":"U", "[31.]":"V","[32.]":"W", "[33.]":"X","[34.]":"Y", "[35.]":"Z"})
img = cv2.imread('result.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img3=cv2.imread('4L.png')
gray3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
rows, cols=gray3.shape
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,180)]
testCells = [np.hsplit(row,(cols/20)) for row in np.vsplit(gray3,(rows/20))]

x = np.array(cells)
y = np.array(testCells)

train = x[:,:50].reshape(-1,400).astype(np.float32)
test= x[:,50:100].reshape(-1,400).astype(np.float32)
testCase = y.reshape(-1,400).astype(np.float32)

k = np.arange(36)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()


knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(testCase,k=5)

numStr=""
for i in result:
 numStr+=letters[str(i)]+" "
print (numStr)

cv2.imshow("",img3)
cv2.waitKey(0)



# matches = result==test_labels
# correct = np.count_nonzero(matches)
# accuracy = (correct/result.size)*100