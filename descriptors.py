# Import statement
import pandas as pd
from rdkit import Chem
from mordred import descriptors

# Locate and load the data base file from the local machine
df = pd.read_csv(r"C:\Users\A K SINGH\Documents\Olfaction Project\odor_data.csv")

# Preview Dataframe
features = df.columns
print(features)
print(df.shape)

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
# d15a = descriptors.CPSA.PNSA()
# d15b = descriptors.CPSA.PPSA()
# d15c = descriptors.CPSA.DPSA()
# d15d = descriptors.CPSA.FNSA()
# d16 = descriptors.DetourMatrix.DetourMatrix()
d17 = descriptors.DistanceMatrix.DistanceMatrix()
d18 = descriptors.EccentricConnectivityIndex.EccentricConnectivityIndex()
d19 = descriptors.EState.AtomTypeEState()
d20a = descriptors.ExtendedTopochemicalAtom.EtaCoreCount()
d20b = descriptors.ExtendedTopochemicalAtom.EtaShapeIndex()
d20c = descriptors.ExtendedTopochemicalAtom.EtaVEMCount()
d21 = descriptors.FragmentComplexity.FragmentComplexity()
d22 = descriptors.Framework.Framework()
# d23a = descriptors.GeometricalIndex.Diameter3D()
# d23b = descriptors.GeometricalIndex.GeometricalShapeIndex()
# d24 = descriptors.GravitationalIndex.GravitationalIndex()
d25a = descriptors.HydrogenBond.HBondAcceptor()
d25b = descriptors.HydrogenBond.HBondDonor()
d26 = descriptors.InformationContent.InformationContent()
# d27a = descriptors.KappaShapeIndex.KappaShapeIndex1()
# d27b = descriptors.KappaShapeIndex.KappaShapeIndex2()
# d27c = descriptors.KappaShapeIndex.KappaShapeIndex3()
d28 = descriptors.Lipinski.Lipinski()
d29 = descriptors.McGowanVolume.McGowanVolume()
d30 = descriptors.MoeType.LabuteASA()
# d31 = descriptors.MolecularDistanceEdge.MolecularDistanceEdge()
d32 = descriptors.MolecularId.MolecularId()
# d33 = descriptors.MomentOfInertia.MomentOfInertia()
# d34 = descriptors.MoRSE.MoRSE()
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
# Creating easy referencing of descriptors

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


