from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from lineardiffusion import LinearDiffusion
from itertools import chain

(immaginitraining, etichettetraining), (immaginitest, etichettetest) = mnist.load_data()
immaginitraining = immaginitraining[:70000]
immaginitest = immaginitest[:70000]
etichettetraining = [str(x) for x in etichettetraining][:70000]
etichettetest = [str(x) for x in etichettetest][:70000]
tutteimmagini = np.concatenate([immaginitraining, immaginitest])
tutteetichette = [str(x) for x in np.concatenate([etichettetraining, etichettetest])]
diffusionelineare = LinearDiffusion()
diffusionelineare.fit(tutteetichette, tutteimmagini)
righe = 10
colonne = 5
fig, ax = plt.subplots(righe, colonne, facecolor='white', figsize=(3, 9))
etichettepredizioni = list(chain.from_iterable([[str(i)]*5 for i in range(10)]))
immaginipredizioni = diffusionelineare.predict(etichettepredizioni,seed=555)
for i in range(righe*colonne):
    ax[i//colonne][i%colonne].imshow(immaginipredizioni[i],
          cmap='gray_r')
    ax[i//colonne][i%colonne].axis('off')
    ax[i//colonne][i%colonne].set_title(f"\"{etichettepredizioni[i]}\"")
fig.suptitle("Images Generated from Prompt")
plt.show()
