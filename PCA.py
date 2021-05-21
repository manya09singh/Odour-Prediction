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
pd.set_option ('display.max_rows', 500)

# Locate and load the data base file from the local machine
df = pd.read_csv(r"C:\Users\A K SINGH\Documents\Olfaction Project\five_c.csv")

# Preview Dataframe
features = ['Common name', 'Odor Class', 'ABCIndex', 'Acidic Group Count', 'Basic Group Count',
            'Adjacency Matrix', 'Aromatic Atom Count', 'Aromatic Bond Count', 'Atom Count', 'Autocorrelation ATS',
            'Autocorrelation ATSC', 'BalabanJ', 'Barysz Matrix ', 'BCUT', 'BertzCT', 'BondCount', 'CarbonTypes', 'Chi',
            'ConstitutionalSum', 'ConstitutionalMean', 'DistanceMatrix', 'EccentricConnectivityIndex', 'AtomTypeEState',
            'EtaCoreCount', 'EtaShapeIndex', 'EtaVEMCount', 'FragmentComplexity', 'Framework', 'HBondAcceptor',
            'HBondDonor', 'InformationContent', 'VMcGowan', 'LabuteASA', 'MID', 'PathCount', 'APol', 'Bpol',
            'RingCount', 'nRot', 'SLogP', 'JGT10', 'Diameter', 'TopoShapeIndex', 'TopoPSA', 'Vabc', 'VAdjMat',
            'MWC01', 'MW', 'WPath', 'Zagreb1']
print(features)

x = df.iloc[:, 6:53]
print(x)
x=StandardScaler().fit_transform(x)
y = df['Odor Class']

#Applying PCA
pca = PCA(n_components=2)
PC = pca.fit_transform(x)
principalDF = pd.DataFrame(data=PC, columns=['pc1', 'pc2'])
print(principalDF)
finalDf = pd.concat([principalDF, df[['Odor Class']]], axis=1)
finalDf
# finalDf.to_csv(r'C:\Users\A K SINGH\Documents\Olfaction Project\PC.csv')
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

fig = ex.scatter(x=loadingdf['PC1'],y=loadingdf['PC2'],text=loadingdf["variable"])
fig.update_layout( height=600,width=500, title_text='loadings plot')
fig.update_traces(textposition='bottom center')
fig.add_shape(type="line", x0=-0, y0=-0.5,x1=-0,y1=2.5, line=dict(color="RoyalBlue",width=3))
fig.add_shape(type="line", x0=-1, y0=0,x1=1,y1=0, line=dict(color="RoyalBlue", width=3))
fig.show()
