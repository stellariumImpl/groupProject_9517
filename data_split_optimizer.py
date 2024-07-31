import os
import numpy as np
from PIL import Image, ImageEnhance
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def collect_grayscale_distributions(root_dirs):
    distributions = defaultdict(list)
    image_distributions = {}
    
    for root_dir in root_dirs:
        index_label_dir = os.path.join(root_dir, 'indexLabel')
        image_dir = os.path.join(root_dir, 'image')
        
        if not os.path.exists(index_label_dir) or not os.path.exists(image_dir):
            continue
        
        for filename in os.listdir(index_label_dir):
            if filename.endswith('.png'):
                label_path = os.path.join(index_label_dir, filename)
                image_path = os.path.join(image_dir, filename)
                
                if not os.path.exists(image_path):
                    continue
                
                try:
                    img = Image.open(label_path).convert('L')
                    img_np = np.array(img)
                    
                    mask = np.isin(img_np, [0, 1, 17], invert=True)
                    valid_pixels = img_np[mask]
                    
                    if valid_pixels.size > 0:
                        unique, counts = np.unique(valid_pixels, return_counts=True)
                        total_valid_pixels = valid_pixels.size
                        
                        img_distribution = {}
                        for gray_value, count in zip(unique, counts):
                            proportion = count / total_valid_pixels
                            distributions[gray_value].append(proportion)
                            img_distribution[gray_value] = proportion
                        
                        image_distributions[image_path] = (img_distribution, label_path)
                except Exception:
                    pass
    
    return distributions, image_distributions

def get_main_classes(image_distributions, min_threshold=0.05, max_threshold=0.2):
    main_classes = {}
    class_counts = defaultdict(int)
    total_images = len(image_distributions)
    
    for _, (dist, _) in image_distributions.items():
        for cls in dist.keys():
            class_counts[cls] += 1
    
    rare_classes = {cls for cls, count in class_counts.items() if count < total_images * 0.05}
    
    for image_path, (dist, _) in image_distributions.items():
        significant_classes = [cls for cls, prop in dist.items() if (cls in rare_classes and prop >= min_threshold) or prop >= max_threshold]
        if not significant_classes:
            significant_classes = [max(dist, key=dist.get)]
        main_classes[image_path] = tuple(significant_classes)
    
    return main_classes

def augment_image(image_path, label_path):
    image = Image.open(image_path)
    label = Image.open(label_path)
    augmented_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_label = label.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(augmented_image)
    augmented_image = enhancer.enhance(1.2)
    return augmented_image, augmented_label

def augment_rare_classes(image_distributions, main_classes, threshold=10):
    augmented_data = {}
    class_counts = defaultdict(int)
    
    for img, classes in main_classes.items():
        for cls in classes:
            class_counts[cls] += 1
    
    for cls, count in class_counts.items():
        if count < threshold:
            for img in [img for img, classes in main_classes.items() if cls in classes]:
                aug_image, aug_label = augment_image(img, image_distributions[img][1])
                aug_image_path = img.replace('.png', '_aug.png')
                aug_label_path = image_distributions[img][1].replace('.png', '_aug.png')
                aug_image.save(aug_image_path)
                aug_label.save(aug_label_path)
                augmented_data[aug_image_path] = (image_distributions[img][0], aug_label_path)
    
    image_distributions.update(augmented_data)
    return image_distributions

def stratified_split(image_distributions, train_ratio=0.7, valid_ratio=0.2):
    main_classes = get_main_classes(image_distributions)
    all_classes = set(cls for classes in main_classes.values() for cls in classes)
    
    train_set, valid_set, test_set = [], [], []
    class_to_images = defaultdict(list)
    
    for img, classes in main_classes.items():
        for cls in classes:
            class_to_images[cls].append(img)
    
    for cls in all_classes:
        cls_images = class_to_images[cls]
        random.shuffle(cls_images)
        
        n_train = max(int(len(cls_images) * train_ratio), 1)
        n_valid = max(int(len(cls_images) * valid_ratio), 1)
        
        train_set.extend((img, image_distributions[img][1]) for img in cls_images[:n_train])
        valid_set.extend((img, image_distributions[img][1]) for img in cls_images[n_train:n_train+n_valid])
        test_set.extend((img, image_distributions[img][1]) for img in cls_images[n_train+n_valid:])
    
    return list(set(train_set)), list(set(valid_set)), list(set(test_set))

def save_split_results(train_set, valid_set, test_set, csv_files):
    os.makedirs(os.path.dirname(csv_files['train']), exist_ok=True)
    
    pd.DataFrame(train_set, columns=['image', 'label']).to_csv(csv_files['train'], index=False)
    pd.DataFrame(valid_set, columns=['image', 'label']).to_csv(csv_files['valid'], index=False)
    pd.DataFrame(test_set, columns=['image', 'label']).to_csv(csv_files['test'], index=False)

if __name__ == "__main__":
    root_dirs = [
        os.path.join('..', 'data', 'WildScenes', 'WildScenes2d', 'V-01'),
        os.path.join('..', 'data', 'WildScenes', 'WildScenes2d', 'V-02'),
        os.path.join('..', 'data', 'WildScenes', 'WildScenes2d', 'V-03')
    ]

    _, image_distributions = collect_grayscale_distributions(root_dirs)
    image_distributions = augment_rare_classes(image_distributions, get_main_classes(image_distributions))
    train_set, valid_set, test_set = stratified_split(image_distributions)

    _data_list_dir = os.path.join('datasets', 'data_list')
    _csv_files = {
        'train': os.path.join(_data_list_dir, 'train.csv'),
        'valid': os.path.join(_data_list_dir, 'valid.csv'),
        'test': os.path.join(_data_list_dir, 'test.csv'),
    }

    save_split_results(train_set, valid_set, test_set, _csv_files)
