import os

from PIL import Image
import imagehash

from tqdm.notebook import tqdm

def group_similar_images(ordered_dir, cutoff, window):
    image_locs = [f"{ordered_dir}/{file_name}" for file_name in os.listdir(ordered_dir) if file_name.endswith(".jpg")]
    groups = [[image_locs[0]]]
    
    for image_loc in tqdm(image_locs):
        img_hash = imagehash.average_hash(Image.open(image_loc))
        
        closest_group_idx = -1
        closest_group_diff = 65
        
        for offset, group in enumerate(groups[-window:][::-1]):
            group_idx = len(groups) - offset - 1
            last_hash = imagehash.average_hash(Image.open(group[-1]))
            diff = img_hash - last_hash
            
            if diff < closest_group_diff:
                closest_group_idx = group_idx
                closest_group_diff = diff
        
        if closest_group_diff <= cutoff:
            groups[closest_group_idx].append(image_loc)
        else:
            groups.append([image_loc])
    
    return groups