import numpy as np
# Defined 3 points in 2D-space:
X=np.array([[2, 1, 0],[4, 3, 0]])
#R = np.cov(X)
R = np.matmul(X,X.T)/3;
# Calculate the SVD decomposition and new basis vectors:
[U,D,V]=np.linalg.svd(R)  # call SVD decomposition
u1=U[:,0] # new basis vectors
u2=U[:,1]
# Calculate the coordinates in new orthonormal basis:
z1=np.matmul(np.transpose(X),u1)
z2=np.matmul(np.transpose(X),u2)
# Calculate the approximation of the original from new basis
Xp1=np.matmul(u1[:,None],z1[None,:])
Xp2=np.matmul(u2[:,None],z2[None,:])
print(Xp1[:,None]+Xp2[:,None])
#print(Xp1+Xp2)
#print(z1[None,:])
#print(z2[None,:])
# Load Iris dataset as in the last PC lab:
from sklearn.datasets import load_iris
iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[:])
     
# We have 4 dimensions of data, plot the first three colums in 3D
X=iris.data
y=iris.target
import matplotlib.pyplot as plt
axes1=plt.axes(projection='3d')
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
plt.show

# Pre-processing is an important step, you can try either StandardScaler (zero mean, unit variance of features)
# or MinMaxScaler (to interval from 0 to 1)
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)

# define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print(pca.get_covariance())
# you can plot the transformed feature space in 3D:
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show

pca.explained_variance_
pca.explained_variance_ratio_

# plot the principal components in 2D, mark different targets in color
plt.scatter(Xpca[y==0, 0], Xpca[y==0, 1], color='red', label='Class 0')
plt.scatter(Xpca[y==1, 0], Xpca[y==1, 1], color='blue', label='Class 1')
plt.scatter(Xpca[y==2, 0], Xpca[y==2, 1], color='green', label='Class 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Split X (original) into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train,y_train)
Ypred=knn1.predict(X_test)
# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix(y_test,Ypred)
ConfusionMatrixDisplay.from_predictions(y_test,Ypred)

#Now do the same (data set split, KNN, confusion matrix), but for PCA-transformed data
# Compare the results with full dataset
from sklearn.model_selection import train_test_split
# Split X (original) into train and test
X_train, X_test, y_train, y_test = train_test_split(Xpca, y, test_size=0.3, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train,y_train)
Ypred=knn1.predict(X_test)
# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix(y_test,Ypred)
ConfusionMatrixDisplay.from_predictions(y_test,Ypred)

# Now do the same, but use only 2-dimensional data of original X (first two columns)
from sklearn.model_selection import train_test_split
# Split X (original) into train and test
#X_train, X_test, y_train, y_test = train_test_split(X[:,0:2], y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(Xpca[:,0:2], y, test_size=0.3, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train,y_train)
Ypred=knn1.predict(X_test)
# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix(y_test,Ypred)
ConfusionMatrixDisplay.from_predictions(y_test,Ypred)

