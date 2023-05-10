import numpy as np

cll = np.load('CLL.npy', allow_pickle=True) #32,10,10

print(cll.shape)

"""
dati = []
for L in np.arange(0,32,1):
    for A in np.arange(0,3,1):
        for i in np.arange(0,10,1):
            for j in np.arange(0,10,1):
                if A == 0:
                    if i < j: None
                    else:
                        dati.append(cll[i][j][L])
                if A == 1:
                    dati.append(cgl[i][j][L])
                if A == 2:
                    if i < j: None
                    else:
                        dati.append(cgg[i][j][L])

dat = np.array(dati)
print(dat.shape)
np.savez('dati_SKA_withbeta', dat)
"""
