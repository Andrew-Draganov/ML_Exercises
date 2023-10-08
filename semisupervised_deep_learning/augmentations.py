from utils import np_onehot
import numpy as np
import torch

# Take top-left and bottom-right quarters from one image and top-right and bottom-left quarters from another
def collage(batch, i, j):
    """ STUDENTS CODE """
    im_size = int(batch.shape[2])
    assert im_size == int(batch.shape[3])
    
    result = np.copy(batch[i])
    interpolation = 0.5
    result[:, :int(im_size/2), :int(im_size/2)] = batch[j, :, :int(im_size/2), :int(im_size/2)]
    result[:, int(im_size/2):, int(im_size/2):] = batch[j, :, int(im_size/2):, int(im_size/2):]
    return result, interpolation

# Linearly interpolate between two images
def mixup(batch, i, j, alpha=0.3):
    """ STUDENTS CODE """
    interpolation = np.random.beta(alpha, alpha)
    result = interpolation * batch[i] + (1 - interpolation) * batch[j]
    return result, interpolation

def no_aug(batch, i, j):
    return batch[i], 1

AUGMENTATION_DICT = {
    'mixup': mixup,
    'collage': collage,
    'no_aug': no_aug
}

def augment(augmentation, batch, labels, n_classes, transform=None):
    batch_size = len(batch)
    new_batch = np.zeros_like(batch)
    merge_indices = np.zeros(batch_size, dtype=np.int32)
    interpolations = np.zeros(batch_size)
    for i, image in enumerate(batch):
        merge_indices[i] = np.random.choice(np.delete(np.arange(batch_size), i))        
        new_batch[i], interpolations[i] = augmentation(batch, i, merge_indices[i])
        
    images = torch.tensor(new_batch)
    interpolations = np.expand_dims(interpolations, 1) # for correct broadcasting
    labels = interpolations * np_onehot(labels, n_classes)
    labels += (1 - interpolations) * np_onehot(merge_indices, n_classes)
    labels = torch.tensor(labels)

    if transform:
        images = transform(images)
        
    return images, labels

