import numpy as np
import cv2
from sklearn.metrics import confusion_matrix


letters = dict({"[0.]":"0", "[1.]":"1","[2.]":"2", "[3.]":"3","[4.]":"4", "[5.]":"5","[6.]":"6", "[7.]":"7","[8.]":"8", "[9.]":"9",
    "[10.]":"A", "[11.]":"B","[12.]":"C", "[13.]":"D","[14.]":"E", "[15.]":"F","[16.]":"G", "[17.]":"H","[18.]":"I", "[19.]":"J"
                 ,"[20.]":"K", "[21.]":"L","[22.]":"M", "[23.]":"N","[24.]":"O", "[25.]":"P","[26.]":"Q", "[27.]":"R","[28.]":"S",
                "[29.]":"T" ,"[30.]":"U", "[31.]":"V","[32.]":"W", "[33.]":"X","[34.]":"Y", "[35.]":"Z"})
img = cv2.imread('result.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img3=cv2.imread('shuffledWrong.png')
gray3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
rows, cols=gray3.shape
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,180)]
testCells = [np.hsplit(row,(cols/20)) for row in np.vsplit(gray3,(rows/20))]

x = np.array(cells)
y = np.array(testCells)
train = x[:,:80].reshape(-1,400).astype(np.float32)
test= x[:,80:100].reshape(-1,400).astype(np.float32)
testCase = y.reshape(-1,400).astype(np.float32)

# for row in test:
#  for col in row:   ## must remove . reshape .astype from test in order to appear pics
#   print(col)
  # cv2.imshow("g",col)
  # cv2.waitKey(0)

k = np.arange(36)
train_labels = np.repeat(k,400)[:,np.newaxis]
test_labels = np.repeat(k,100)[:,np.newaxis]


knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)

numStr=""
for i in result:
 numStr+=letters[str(i)]+" "
print (numStr)

# cv2.imshow("",img3)
# cv2.waitKey(0)

print ('\n Confusion Matrix: \n')
confusion_matrix_output =confusion_matrix(test_labels, result)
print (end='  ')
for i in letters:
    print (letters[str(i)] ,end=' ')
print('\n')

for i,x in zip(confusion_matrix_output,letters):
 print(letters[str(x)], end=' ')
 for j in i:
  print (j, end=' ')
 print ('\n')

sum=0
for i in range (len(confusion_matrix_output)):
  for j in range(len(confusion_matrix_output[0])):
    if i==j:
     sum+=confusion_matrix_output[i][j]

print ("accuracy: ", (sum/result.size) *100)


