# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 08:41:28 2021

@author: diego
"""
random_indices = np.random.randint(0, nb_images, 25)
plt.figure(figsize=(10,10))

for iCpt, iIdx in enumerate(random_indices):
    plt.subplot(5, 5, iCpt)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_train[iIdx][:, :, 0], cmap=plt.cm.binary)
    plt.xlabel(labels_train[iIdx]);

plt.show()