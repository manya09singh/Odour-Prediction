# Odour-Prediction

The full length explanation of piece by piece code can be found here : https://medium.com/nerd-for-tech/olfactory-prediction-unsupervised-learning-f9a19f4154d1

The code below works on the conept of clustering data to reach categorized data, of different odors
The input data we deal with has the molecular descriptor values of different odor groups. It is a numerical representaion of odor, or so to say.
This can be calculated directly from the SMILE or molecular formula of the compound.

This is then used to train a K-means clustering algorithm, which gives us a scatter plot of all the compounds.
Later a mapping is developed between the odor groups classified and the originally mapped data to check it's accuracy


First, we need to extract the molecular descriptors of all compounds in the dataset
```
# Import statement
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as ex
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from rdkit import Chem
from mordred import descriptors

# Locate and load the data base file from the local machine
df = pd.read_csv(r"C:\Olfaction Project\odor_data.csv")

# Preview Dataframe
features = df.columns
# Define all types of descriptors
d = []
d01 = descriptors.ABCIndex.ABCIndex()
d02a = descriptors.AcidBase.AcidicGroupCount()
d02b = descriptors.AcidBase.BasicGroupCount()
d03 = descriptors.AdjacencyMatrix.AdjacencyMatrix(type="VE2")
d04a = descriptors.Aromatic.AromaticAtomsCount()
d04b = descriptors.Aromatic.AromaticBondsCount()
d05 = descriptors.AtomCount.AtomCount()
d06a = descriptors.Autocorrelation.ATS()
d06b = descriptors.Autocorrelation.ATSC()
d07 = descriptors.BalabanJ.BalabanJ()
d08 = descriptors.BaryszMatrix.BaryszMatrix()
d09 = descriptors.BCUT.BCUT()
d10 = descriptors.BertzCT.BertzCT()
d11 = descriptors.BondCount.BondCount()
d12 = descriptors.CarbonTypes.CarbonTypes()
d13 = descriptors.Chi.Chi()
d14a = descriptors.Constitutional.ConstitutionalSum()
d14b = descriptors.Constitutional.ConstitutionalMean()
d17 = descriptors.DistanceMatrix.DistanceMatrix()
d18 = descriptors.EccentricConnectivityIndex.EccentricConnectivityIndex()
d19 = descriptors.EState.AtomTypeEState()
d20a = descriptors.ExtendedTopochemicalAtom.EtaCoreCount()
d20b = descriptors.ExtendedTopochemicalAtom.EtaShapeIndex()
d20c = descriptors.ExtendedTopochemicalAtom.EtaVEMCount()
d21 = descriptors.FragmentComplexity.FragmentComplexity()
d22 = descriptors.Framework.Framework()
d25a = descriptors.HydrogenBond.HBondAcceptor()
d25b = descriptors.HydrogenBond.HBondDonor()
d26 = descriptors.InformationContent.InformationContent()
d28 = descriptors.Lipinski.Lipinski()
d29 = descriptors.McGowanVolume.McGowanVolume()
d30 = descriptors.MoeType.LabuteASA()
d32 = descriptors.MolecularId.MolecularId()
d35 = descriptors.PathCount.PathCount()
d36a = descriptors.Polarizability.APol()
d36b = descriptors.Polarizability.BPol()
d37 = descriptors.RingCount.RingCount()
d38 = descriptors.RotatableBond.RotatableBondsCount()
d39 = descriptors.SLogP.SLogP()
d40 = descriptors.TopologicalCharge.TopologicalCharge()
d41a = descriptors.TopologicalIndex.Diameter()
d41b = descriptors.TopologicalIndex.TopologicalShapeIndex()
d42 = descriptors.TopoPSA.TopoPSA()
d43 = descriptors.VdwVolumeABC.VdwVolumeABC()
d44 = descriptors.VertexAdjacencyInformation.VertexAdjacencyInformation()
d45 = descriptors.WalkCount.WalkCount()
d46 = descriptors.Weight.Weight()
d47 = descriptors.WienerIndex.WienerIndex()
d48 = descriptors.ZagrebIndex.ZagrebIndex()


d = [d01, d02a, d02b, d03, d04a, d04b, d05, d06a, d06b, d07, d08,
     d09, d10, d11, d12, d13, d14a, d14b,
     d17, d18, d19, d20a, d20b, d20c, d21, d22,
     d25a, d25b, d26, d28, d29, d30, d32,
     d35, d36a, d36b, d37, d38, d39, d40, d41a, d41b, d42, d43, d44,
     d45, d46, d47, d48]

chemicals = df["SMILE"]
print(chemicals)
mol = []
for ele in chemicals:
    temp = Chem.MolFromSmiles(ele)
    mol.append(temp)

for desc in d:
    for m in mol:
        print(desc(m))



# The result of this can be either exported or transferred to a new csv file
Further analysis will be performed on this updated dataset

```

## PCA: Reducing the number of decriptors
```
features = ['Common name', 'Odor Class', 'ABCIndex', 'Acidic Group Count', 'Basic Group Count',
            'Adjacency Matrix', 'Aromatic Atom Count', 'Aromatic Bond Count', 'Atom Count', 'Autocorrelation ATS',
            'Autocorrelation ATSC', 'BalabanJ', 'Barysz Matrix ', 'BCUT', 'BertzCT', 'BondCount', 'CarbonTypes', 'Chi',
            'ConstitutionalSum', 'ConstitutionalMean', 'DistanceMatrix', 'EccentricConnectivityIndex', 'AtomTypeEState',
            'EtaCoreCount', 'EtaShapeIndex', 'EtaVEMCount', 'FragmentComplexity', 'Framework', 'HBondAcceptor',
            'HBondDonor', 'InformationContent', 'VMcGowan', 'LabuteASA', 'MID', 'PathCount', 'APol', 'Bpol',
            'RingCount', 'nRot', 'SLogP', 'JGT10', 'Diameter', 'TopoShapeIndex', 'TopoPSA', 'Vabc', 'VAdjMat',
            'MWC01', 'MW', 'WPath', 'Zagreb1']

x = df.iloc[:, 6:53]
x=StandardScaler().fit_transform(x)
y = df['Odor Class']

#Applying PCA
pca = PCA(n_components=2)
PC = pca.fit_transform(x)

#creating a new dataframe with results of PCA appended

principalDF = pd.DataFrame(data=PC, columns=['pc1', 'pc2'])
finalDf = pd.concat([principalDF, df[['Odor Class']]], axis=1)

# Scatter plot of the principal component 1 vs 2

print(pca.explained_variance_ratio_)
plt.scatter(PC[:,0],PC[:,1],c=df['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

PCloadings = pca.components_.T * np.sqrt(pca.explained_variance_)
components=df.columns.tolist()
components=components[6:53]
print(components)
loadingdf=pd.DataFrame(PCloadings,columns=('PC1','PC2'))
loadingdf["variable"]=components

# To view which decriptor shows max variance

fig = ex.scatter(x=loadingdf['PC1'],y=loadingdf['PC2'],text=loadingdf["variable"])
fig.update_layout(height=600,width=500, title_text='loadings plot')
fig.update_traces(textposition='bottom center')
fig.add_shape(type="line", x0=-0, y0=-0.5,x1=-0,y1=2.5, line=dict(color="RoyalBlue",width=3))
fig.add_shape(type="line", x0=-1, y0=0,x1=1,y1=0, line=dict(color="RoyalBlue", width=3))
fig.show()
```

## k-means clustering

```
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

# Visualizing the clusters

x_axis = df_new['Component0']
y_axis = df_new['Component1']
plt.figure(figsize = (10,8))
sns.scatterplot(x_axis, y_axis, hue = df_new['Segment K-means PCA'], palette=['g','r','c','m','k','orange']
plt.title("Clusters by PCA components")
plt.show()

```
## Evaluating the Clusters

```
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
```
From the split of datasets I used, I found the lowest RMSE (1.19) and MAE (1.42) values for dataset E, with 8 clusters

