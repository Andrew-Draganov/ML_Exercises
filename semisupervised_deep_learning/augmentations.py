from utils import np_onehot
import numpy as np
import torch

def collage(batch_i, batch_j):
    """
    Take top-left and bottom-right quarters from one image and top-right and bottom-left quarters from another.

    Since this takes one half from one image and another half from another, our class interpolation should be 0.5
    """
    # Needs to be a square image to apply collage augmentation
    im_size = int(batch_i.shape[1])
    assert im_size == int(batch_i.shape[2])
    result = None
    interpolation = None
    
    ### YOUR CODE HERE
    interpolation = 0.5

    result = np.copy(batch_i)
    result[:, :int(im_size/2), :int(im_size/2)] = batch_j[:, :int(im_size/2), :int(im_size/2)]
    result[:, int(im_size/2):, int(im_size/2):] = batch_j[:, int(im_size/2):, int(im_size/2):]
    ### END CODE

    return result, interpolation

def mixup(batch_i, batch_j, alpha=0.3):
    """
    Linearly interpolate between two images.
    We find the interpolation value in [0, 1] for you by sampling from a Beta distribution; see https://arxiv.org/pdf/1710.09412.pdf
    """
    interpolation = np.random.beta(alpha, alpha)
    result = None

    ### YOUR CODE HERE
    result = interpolation * batch_i + (1 - interpolation) * batch_j
    ### END CODE

    return result, interpolation

def no_aug(batch_i, batch_j):
    """ STUDENTS CODE """
    return batch_i, batch_j

AUGMENTATION_DICT = {
    'mixup': mixup,
    'collage': collage,
    'no_aug': no_aug
}

def augment(augmentation, batch, labels, n_classes):
    """
    Apply the augmentation to the batch in question.

    The mixup and collage augmentations will combine two images to make something 'in between' them.
    The label should then be a linear combination of the two image labels.
    """
    if augmentation is None:
        augmentation = no_aug

    batch_size = len(batch)
    new_batch = np.zeros_like(batch)
    merge_indices = np.zeros(batch_size, dtype=np.int32)
    interpolations = np.zeros(batch_size)
    for i, image in enumerate(batch):
        # merge_indices[i] is the index of the element that batch[i] will be augmented with
        # We do not allow merge_indices[i] = i
        merge_indices[i] = np.random.choice(np.delete(np.arange(batch_size), i))        

        # Apply the augmentation
        new_batch[i], interpolations[i] = augmentation(batch[i], batch[merge_indices[i]])
        
    images = torch.tensor(new_batch)
    interpolations = np.expand_dims(interpolations, 1) # for correct broadcasting
    labels = interpolations * np_onehot(labels, n_classes)
    labels += (1 - interpolations) * np_onehot(merge_indices, n_classes)
    labels = torch.tensor(labels)

    return images, labels


def test_augmentations():
    pass


if __name__ == '__main__':
    test_augmentations()
