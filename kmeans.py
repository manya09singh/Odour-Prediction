import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt


kmeans_pca = KMeans(n_clusters=6, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)

df_new = pd.concat([df.reset_index(drop= True), pd.DataFrame(scores_pca)], axis=1)
df_new.columns.values[-5:] = ['Component0','Component1','Component2','Component3','Component4']
df_new['Segment K-means PCA'] = kmeans_pca.labels_

df_new['Segment'] = df_new['Segment K-means PCA'].map({0:'first',
                                                      1:'second',
                                                      2:'third',
                                                      3:'fourth',
                                                      4:'fifth',
                                                       5:'sixth',
                                                       6:'seventh',
                                                       7:'eighth'})
print(df_new.head())
# Visualizing the clusters

x_axis = df_new['Component0']
y_axis = df_new['Component1']
plt.figure(figsize = (10,8))
sns.scatterplot(x_axis, y_axis, hue = df_new['Segment K-means PCA'], palette=['g','r','c','m','k','orange'])
# sns.scatterplot(x_axis, y_axis, hue = df_new['Segment K-means PCA'], palette=['g','r','c','m','lightcoral','orange','brown','lawngreen','orchid'])
plt.title("Clusters by PCA components")
plt.show()


# df.to_csv(r'C:\Users\A K SINGH\Documents\Olfaction Project\Results\pcafive_c_new.csv', index = False)
df_new.to_csv(r'C:\Users\A K SINGH\Documents\Olfaction Project\Results\kmeansfive_a_6clusters.csv', index = False)
# define predictor and response variables
X = df_new[['target']]
y = df_new['Segment K-means PCA']



# define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# build multiple linear regression model
model = LinearRegression()

# use k-fold CV to evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

# view mean absolute error

print("MAE")
print(mean(absolute(scores)))
print("RMSE")
print(sqrt(mean(absolute(scores))))
