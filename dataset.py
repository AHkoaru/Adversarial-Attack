import os
import numpy as np
from PIL import Image

class CitySet:
    def __init__(self, dataset_dir=None, images=None, gt_images=None):
        """
        Load and convert Cityscapes dataset images to NumPy arrays.
        
        Args:
            dataset_dir (str, optional): Path to Cityscapes dataset directory
                             (e.g. "Projects/mmsegmentation/datasets/cityscapes")
                             
                             Cityscapes dataset directory structure:
                             cityscapes/
                             ├── leftImg8bit/
                             │   ├── train/
                             │   │   ├── aachen/
                             │   │   │   ├── aachen_000000_000019_leftImg8bit.png
                             │   │   │   └── ...
                             │   │   ├── bochum/
                             │   │   └── ...
                             │   ├── val/
                             │   └── test/
                             └── gtFine/
                                 ├── train/
                                 │   ├── aachen/
                                 │   │   ├── aachen_000000_000019_gtFine_labelIds.png
                             │   │   └── ...
                             │   ├── bochum/
                             │   └── ...
                             ├── val/
                             └── test/
                                 
            images (list, optional): List of images as NumPy arrays
        """
        if dataset_dir is not None:
            # Verify Cityscapes dataset structure
            leftimg_dir = os.path.join(dataset_dir, "leftImg8bit")
            if not os.path.exists(leftimg_dir):
                raise ValueError(f"Invalid Cityscapes dataset structure. {leftimg_dir} does not exist.")
                
            self.dataset_dir = os.path.join(dataset_dir, "leftImg8bit", "val")
            if not os.path.exists(self.dataset_dir):
                raise ValueError(f"Validation dataset not found at: {self.dataset_dir}")
                
            self.gt_dir = os.path.join(dataset_dir, "gtFine", "val")
            if not os.path.exists(self.gt_dir):
                raise ValueError(f"Ground truth directory not found at: {self.gt_dir}")
                
            self.images, self.filenames, self.gt_images = self._load_images()
        elif images is not None:
            self.images = images
            # Generate filenames as image_0.png, image_1.png, etc.
            self.filenames = [f"image_{i}.png" for i in range(len(images))]
            # For images provided directly, we don't have ground truth
            if gt_images is not None:
                self.gt_images = gt_images
            else:
                self.gt_images = [None] * len(images)
        else:
            raise ValueError("Either dataset_dir or images must be provided")
    
    def _load_images(self):
        """
        Load all images from Cityscapes dataset and convert to list of NumPy arrays.
        Images are converted to BGR format.
        
        Returns:
            tuple: (list of images as NumPy arrays in BGR format, list of corresponding filenames, list of ground truth images)
        """
        images = []
        filenames = []
        gt_images = []
        
        # Iterate through city directories in Cityscapes
        for city in os.listdir(self.dataset_dir):
            city_path = os.path.join(self.dataset_dir, city)
            
            if os.path.isdir(city_path):
                # Only load Cityscapes format images (ending with _leftImg8bit.png)
                for filename in os.listdir(city_path):
                    if filename.endswith("_leftImg8bit.png"):
                        file_path = os.path.join(city_path, filename)
                        
                        try:
                            # Load image and convert to NumPy array
                            img = Image.open(file_path)
                            img_array = np.array(img)
                            
                            # RGB -> BGR conversion (Cityscapes stores in RGB)
                            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                                img_array = img_array[:, :, ::-1]
                            
                            images.append(img_array)
                            filenames.append(os.path.join(city, filename))  # Store as city/filename
                            
                            # Load corresponding ground truth
                            gt_filename = filename.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
                            gt_path = os.path.join(self.gt_dir, city, gt_filename)
                            gt_img = Image.open(gt_path)
                            gt_array = np.array(gt_img)
                            gt_images.append(gt_array)
                                
                        except Exception as e:
                            print(f"Error loading image ({city}/{filename}): {e}")
        
        if not images:
            print("Warning: No images loaded. Please check dataset path.")
            
        return images, filenames, gt_images
    
    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Return the image, filename and ground truth at the specified index.
        
        Args:
            idx (int): Dataset index
            
        Returns:
            tuple: (image as NumPy array, filename, ground truth as NumPy array)
        """
        return self.images[idx], self.filenames[idx], self.gt_images[idx]
    
class ADESet:
    def __init__(self, dataset_dir=None, images=None):
        """
        Load and convert ADE20K dataset images to NumPy arrays.
        
        Args:
            dataset_dir (str, optional): Path to ADE20K dataset directory
                             (e.g. "Projects/mmsegmentation/datasets/ade")
                             
                             ADE20K dataset directory structure:
                             ade20k/
                             ├── images/
                             │   ├── validation/
                             │   │   ├── ADE_val_00000001.png
                             │   │   └── ...
                             │   └── training/
                             └── annotations/
                                 ├── validation/
                                 │   ├── ADE_val_00000001.png
                                 │   └── ...
                                 └── training/
                                 
            images (list, optional): List of images as NumPy arrays
        """
        if dataset_dir is not None:
            # Verify ADE20K dataset structure
            images_dir = os.path.join(dataset_dir, "images")
            if not os.path.exists(images_dir):
                raise ValueError(f"Invalid ADE20K dataset structure. {images_dir} does not exist.")
                
            self.dataset_dir = os.path.join(dataset_dir, "images", "validation")
            if not os.path.exists(self.dataset_dir):
                raise ValueError(f"Validation dataset not found at: {self.dataset_dir}")
                
            self.gt_dir = os.path.join(dataset_dir, "annotations", "validation")
            if not os.path.exists(self.gt_dir):
                raise ValueError(f"Ground truth directory not found at: {self.gt_dir}")
                
            self.images, self.filenames, self.gt_images = self._load_images()
        elif images is not None:
            self.images = images
            # Generate filenames as image_0.png, image_1.png, etc.
            self.filenames = [f"image_{i}.png" for i in range(len(images))]
            # For images provided directly, we don't have ground truth
            self.gt_images = [None] * len(images)
        else:
            raise ValueError("Either dataset_dir or images must be provided")
    
    def _load_images(self):
        """
        Load all images from ADE20K dataset and convert to list of NumPy arrays.
        Images are converted to BGR format.
        
        Returns:
            tuple: (list of images as NumPy arrays in BGR format, list of corresponding filenames, list of ground truth images)
        """
        images = []
        filenames = []
        gt_images = []
        
        # Load all images in the validation directory
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(self.dataset_dir, filename)
                
                try:
                    # Load image and convert to NumPy array
                    img = Image.open(file_path)
                    img_array = np.array(img)
                    
                    # RGB -> BGR conversion
                    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                        img_array = img_array[:, :, ::-1]
                    
                    images.append(img_array)
                    filenames.append(filename)
                    
                    # Load corresponding ground truth (always .png)
                    gt_filename = os.path.splitext(filename)[0] + ".png"
                    gt_path = os.path.join(self.gt_dir, gt_filename)
                    gt_img = Image.open(gt_path)
                    gt_array = np.array(gt_img)
                    gt_images.append(gt_array)
                        
                except Exception as e:
                    print(f"Error loading image ({filename}): {e}")
        
        if not images:
            print("Warning: No images loaded. Please check dataset path.")
            
        return images, filenames, gt_images
    
    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Return the image, filename and ground truth at the specified index.
        
        Note: The returned image is in BGR format and stored as a NumPy array.
              If you need to save this image, use the save_image() method to ensure proper
              color conversion and to avoid quality loss from JPEG compression.
        
        Args:
            idx (int): Dataset index
            
        Returns:
            tuple: (image as NumPy array in BGR format, filename, ground truth as NumPy array)
        """
        return self.images[idx], self.filenames[idx], self.gt_images[idx]

### test code
if __name__ == "__main__":
    # Test CitySet
    print("Testing CitySet...")
    dataset_dir = "./datasets/cityscapes"
    city_set = CitySet(dataset_dir)
    print(f"Loaded {len(city_set)} images")
    image, filename, gt = city_set[1]
    print(f"Image shape: {image.shape}")
    print(f"Filename: {filename}")
    print(f"Ground truth shape: {gt.shape}")
    
    # Test ADESet
    print("\nTesting ADESet...")
    dataset_dir = "./datasets/ade20k"
    ade_set = ADESet(dataset_dir)
    print(f"Loaded {len(ade_set)} images")
    image, filename, gt = ade_set[1]
    print(f"Image shape: {image.shape}")
    print(f"Filename: {filename}")
    print(f"Ground truth shape: {gt.shape}")
    

    