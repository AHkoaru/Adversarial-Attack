import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

class DataFolderWriter:
    def __init__(self, dataset_name, split, root_dir="data"):
        self.dataset_name = dataset_name
        self.split = split
        self.root_dir = Path(root_dir) / dataset_name / split

    def save_image(self, rel_path, image_np, attack_name, subdir="images"):
        """
        Save adversarial image.
        rel_path: relative path (e.g., 'frankfurt/frankfurt_000000_000294_leftImg8bit.png')
        image_np: numpy array (H, W, 3) RGB or BGR
        """
        p = Path(rel_path)
        # data/<dataset>/<split>/<subdir>/<rel_path_parent>/attack_<name>/<filename>
        save_dir = self.root_dir / subdir / p.parent / f"attack_{attack_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / p.name
        
        # Versioning if file exists
        if save_path.exists():
            stem = p.stem
            suffix = p.suffix
            counter = 1
            while save_path.exists():
                save_path = save_dir / f"{stem}_v{counter}{suffix}"
                counter += 1
                
        # Ensure uint8
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
            
        Image.fromarray(image_np).save(save_path)
        return str(save_path)

    def save_metadata(self, rel_path, metadata, attack_name):
        p = Path(rel_path)
        save_dir = self.root_dir / "attack_meta" / p.parent / f"attack_{attack_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{p.stem}.json"
        
        # Handle non-serializable types
        def default_converter(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            return str(o)

        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=default_converter)
            
        return str(save_path)
