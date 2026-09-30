import os
import numpy as np
import random
import torchvision.datasets as datasets

class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index] 
        data = np.load(path)
        
        moments = data['codes']    
        aug_size = len(moments) 
        rand = random.randint(0, aug_size-1) 
        moments = moments[rand]
        return target, moments

class ImageFolderDict(datasets.ImageFolder):
    def __getitem__(self, index):
        image, class_id = super().__getitem__(index)
        return {
            "image": image,
            "class_id": class_id,
        }