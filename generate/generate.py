import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# verificando o diretório com as imagens
sub_dir = os.listdir('CNN letter Dataset')

images = []
answers = []

# iterando e colocando as imagens
for d in sub_dir:
    img_in_dir = os.listdir(f'CNN letter Dataset/{d}')
    answers.extend([d] * len(img_in_dir)) # adicionando as respostas
    for img in img_in_dir:
        images.append(f'CNN letter Dataset/{d}/{img}')

# gerando e embaralhando os índices
random.seed(6661)
idxs = [i for i in range(len(images))]
random.shuffle(idxs)

X = np.zeros((len(images), 7500))
Y = np.zeros((len(images),), dtype='S1')

print('Processando...')
for i in tqdm(idxs):
    charac = answers[i]
    im = Image.open(images[i])
    imarr = np.array(im)
    flt = imarr.flatten()
    X[i] = flt
    Y[i] = charac

print('Salvando...')
np.save('features.npy', X)
np.save('targets.npy', Y)

print('Finalizado.')