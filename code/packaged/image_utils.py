import cv2

def revert_normalisation(tensor):
    return (tensor.permute(1, 2, 0) + 1) / 2

def revert_normalisation_batch(tensors):
    return (tensors.permute(0, 2, 3, 1) + 1) / 2

def load_rgb_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)