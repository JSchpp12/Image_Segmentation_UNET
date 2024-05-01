import numpy as np
from utils import get_colormap

class Ohe():
    def __init__(self, mask, n_classes):
        self.m_mask_1 = mask
        self.color_dict = get_colormap()
        self.n_classes = n_classes

    def one_hot_encode(self, converted_to_RGB_labels):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : x: List of sample Labels
        : return: Numpy array of one-hot encoded labels
        """
        return np.eye(self.n_classes)[converted_to_RGB_labels].astype(int)
    

    def convert_rgb_to_int(self):
        label_seg = np.zeros((self.m_mask_1.shape[0], self.m_mask_1.shape[1]), dtype=np.uint8)
        def convert_rgb_to_int_helper(color, new_label):
            for a in range(self.m_mask_1.shape[0]):
                for b in range(self.m_mask_1.shape[1]):
                    if np.array_equal(self.m_mask_1[a][b], color[::-1]):
                        label_seg[a][b] = new_label
            return label_seg
        
        i = 0
        for v in self.color_dict.values():
            label_seg = convert_rgb_to_int_helper(v, i)
            i += 1

        return label_seg
    
    def one_hot_encoded_mask(self):
        a = self.convert_rgb_to_int()
    
        b = self.one_hot_encode(a)
        return b
