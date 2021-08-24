import cv2
from .mask import get_mask
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as albu
import matplotlib.pyplot as plt

class DiamondDataset(Dataset):
    """Diamond Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        filenames (str): path with images
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            filenames,
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.filenames = filenames
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read image
        image = cv2.imread(self.filenames[i] + '/' + 'Darkfield_EF.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # extract mask
        mask = get_mask(Path(self.filenames[i]))
        mask = mask.transpose((1, 2, 0))
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.filenames)
    

""" Augmentations and Preprocessing """

def get_train_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.Resize(height=1200, width=1200),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_valid_augmentation():
    test_transform = [
        albu.Resize(height=1200, width=1200),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


""" helper function for data visualization """

def visualize(
    image,
    mask,
    classes = [
        "Crystal", "Cloud", "Feather",
        "Pinpoint", "Needle", "Twinning wisp",
        "Bruise", "Chip", "Surface graining", "Internal graining"
    ]
):
    """PLot image and masks."""
    n = 1 + mask.shape[2]
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, n, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('image'.title())
    plt.imshow(image)
    
    for i in range(1, n):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(classes[i - 1].title())
        plt.imshow(mask[:, :, i - 1])
        
    plt.show()