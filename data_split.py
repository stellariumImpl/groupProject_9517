import pandas as pd
import os
import sklearn
from PIL import Image
import numpy as np
import logging

class WildScenesDataset:
    _data_list_dir = os.path.join('datasets', 'data_list')
    _csv_files = {
        'train': os.path.join(_data_list_dir, 'train.csv'),
        'valid': os.path.join(_data_list_dir, 'valid.csv'),
        'test': os.path.join(_data_list_dir, 'test.csv'),
    }

    # Pixel values are label index values (class indices [0,14] as assigned with classes sorted alphabetically by class name).
    # Define the mapping from label index to trainId
    _label_to_trainid = {
        1: 0,  # Asphalt
        2: 1,  # Bush *
        3: 2,  # Dirt *
        4: 3,  # Fence *
        5: 4,  # Grass *
        6: 5,  # Gravel *
        7: 6,  # Log *
        8: 7,  # Mud *
        9: 8,  # Other-object *
        10: 9,  # Other-terrain *
        11: 10,  # Pole
        12: 11,  # Rock *
        13: 12,  # Sky *
        14: 13,  # Structure *
        15: 14,  # Tree-foliage *
        16: 15,  # Tree-trunk *
        17: 16,  # Vehicle
        18: 17,  # Water *
    }

    def __init__(self, dataset_type, transform=None):
        assert dataset_type in ('train', 'valid', 'test')
        self._dataset_type = dataset_type
        self._data_frame = pd.read_csv(WildScenesDataset._csv_files[self._dataset_type])
        self._transform = transform

    def __len__(self):
        return len(self._data_frame)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")
        try:
            image_path = self._data_frame['image'].iloc[index]
            label_path = self._data_frame['label'].iloc[index]

            image = Image.open(image_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            # 将灰度图转换为Numpy数组
            label_np = np.array(label)

            # 将标签索引映射到训练标识（trainId）值
            label_trainId = np.vectorize(lambda x: self._label_to_trainid.get(x, 255))(label_np)

            if self._transform is not None:
                for t in self._transform:
                    image, label_trainId = t(image, label_trainId)

            return image, label_trainId
        except Exception as e:
            logging.error(f"Error loading item at index {index}: {str(e)}")
            raise

    @staticmethod
    def _get_image_label_dir():
        """
        Traverse server image and label directories and yield image and label paths.
        :return: Generator yielding (image path, label path)
        """
        data_err = 'data error. check!'
        image_base = WildScenesDataset.image_file_base
        label_base = WildScenesDataset.label_file_base

        for image in os.listdir(image_base):
            image_origin = os.path.join(image_base, image)
            image_label = os.path.join(label_base, image)

            if not (os.path.isfile(image_label) and
                    os.path.exists(image_label) and
                    os.path.isfile(image_label)):
                print(image_origin, image_label, data_err)  # Print error message and skip if paths are invalid
                continue

            yield image_origin, image_label

    @staticmethod
    def make_data_list(train_rate=0.7, valid_rate=0.2, shuffle=True):
        """
        Shuffle and generate data_list CSV files with image and label paths sorted by filename.
        :param train_rate: Training set ratio, default 0.7
        :param valid_rate: Validation set ratio, default 0.2
        :param shuffle: Whether to shuffle the dataset, default True
        :return: None
        """
        g = WildScenesDataset._get_image_label_dir()  # Get generator
        abspaths = list(g)  # Convert generator to list

        # Create DataFrame with image and label paths
        df = pd.DataFrame(
            data=abspaths,
            columns=['image', 'label']
        )

        # Sort DataFrame by filename (assumed to be timestamp in a sortable format)
        df['timestamp'] = df['image'].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0].split('-')[0]))
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        if shuffle:
            df = sklearn.utils.shuffle(df)  # Shuffle dataframe if specified

        # Calculate sizes for train, valid, and test sets
        train_size = int(df.shape[0] * train_rate)
        valid_size = int(df.shape[0] * valid_rate)

        print('total: {:d} | train: {:d} | val: {:d} | test: {:d}'.format(
            df.shape[0], train_size, valid_size,
            df.shape[0] - train_size - valid_size))

        # Split dataframe into train, valid, and test sets
        df_train = df[0: train_size]
        df_valid = df[train_size: train_size + valid_size]
        df_test = df[train_size + valid_size:]

        # Save train, valid, and test sets to CSV files
        df_train[['image', 'label']].to_csv(os.path.join(WildScenesDataset.csv['train']), index=False)
        df_valid[['image', 'label']].to_csv(os.path.join(WildScenesDataset.csv['valid']), index=False)
        df_test[['image', 'label']].to_csv(os.path.join(WildScenesDataset.csv['test']), index=False)

    # 测试用
    @staticmethod
    def test_label_mapping(label_path):
        """
        Test the label mapping for a single label image.

        :param label_path: Path to the label image
        :return: Tuple of original label numpy array and mapped trainId numpy array
        """
        # Open the label image and convert to numpy array
        label = Image.open(label_path).convert('L')
        label_np = np.array(label, dtype=np.uint8)

        # Map label indices to trainId values
        label_trainId = np.vectorize(lambda x: WildScenesDataset._label_to_trainid.get(x, 255))(label_np)

        return label_np, label_trainId


if __name__ == '__main__':
    # Example usage
    root_dir = os.path.join('..', 'WildScenes_Dataset-61gd5a0t-', 'data', 'WildScenes', 'WildScenes2d', 'V-01')
    WildScenesDataset.image_file_base = os.path.join(root_dir, 'image')
    WildScenesDataset.label_file_base = os.path.join(root_dir, 'indexLabel')
    WildScenesDataset.make_data_list()

    # Test label mapping 测试labelIndex里的第一张
    test_label_path = os.path.join(WildScenesDataset.label_file_base,
                                   '1623377790-818434554.png')  # Replace with an actual image name
    original_label, mapped_label = WildScenesDataset.test_label_mapping(test_label_path)

    print("Original label array (shape: {}):".format(original_label.shape))
    print(original_label)
    print("\nMapped trainId array (shape: {}):".format(mapped_label.shape))
    print(mapped_label)

    # Optional: print unique values in each array
    print("\nUnique values in original label array:", np.unique(original_label))
    print("Unique values in mapped trainId array:", np.unique(mapped_label))
