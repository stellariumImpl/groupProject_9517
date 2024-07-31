# Drawing here on Ascetics/Pytorch-SegToolbox. But we use another version of data split method in data_split_optimizer.py, this part of code is just the previous one
import pandas as pd
import os
import sklearn
from PIL import Image
import numpy as np
import logging


class WildScenesDataset:
    root_dirs = [
        os.path.join('..', 'data', 'WildScenes', 'WildScenes2d', 'V-01'),
        os.path.join('..', 'data', 'WildScenes', 'WildScenes2d', 'V-02'),
        os.path.join('..', 'data', 'WildScenes', 'WildScenes2d', 'V-03')
    ]
    _data_list_dir = os.path.join('datasets', 'data_list')
    _csv_files = {
        'train': os.path.join(_data_list_dir, 'train.csv'),
        'valid': os.path.join(_data_list_dir, 'valid.csv'),
        'test': os.path.join(_data_list_dir, 'test.csv'),
    }
    csv = _csv_files
    _label_to_trainid = {
        0: 15,  # Background
        1: 16,  # Ignore
        2: 0,  # Bush
        3: 1,  # Dirt
        4: 2,  # Fence
        5: 3,  # Grass
        6: 4,  # Gravel
        7: 5,  # Log
        8: 6,  # Mud
        9: 7,  # Other-Object
        10: 8,  # Other-terrain
        11: 16,  # Ignore
        12: 9,  # Rock
        13: 10,  # Sky
        14: 11,  # Structure
        15: 12,  # Tree-foliage
        16: 13,  # Tree-trunk
        17: 16,  # Ignore
        18: 14,  # Water
    }
    def __init__(self, dataset_type, transform=None):
        # Drawing here on Ascetics/Pytorch-SegToolbox.
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
            # Drawing here on Ascetics/Pytorch-SegToolbox.
            image_path = self._data_frame['image'].iloc[index]
            label_path = self._data_frame['label'].iloc[index]

            image = Image.open(image_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            # Convert grayscale image to Numpy array
            label_np = np.array(label)

            # Map tag index to training identifier (trainId) value
            label_trainId = np.vectorize(lambda x: self._label_to_trainid.get(x, 255))(label_np)

            if self._transform is not None:
                for t in self._transform:
                    image, label_trainId = t(image, label_trainId)

            return image, label_trainId
        except Exception as e:
            logging.error(f"Error loading item at index {index}: {str(e)}")

    @staticmethod
    def _get_image_label_dir():
        """
        Traverse server image and label directories and yield image and label paths.
        :return: Generator yielding (image path, label path)
        """
        # Drawing here on Ascetics/Pytorch-SegToolbox.
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
        # Ensure the directory exists
        data_list_dir = WildScenesDataset._data_list_dir
        if not os.path.exists(data_list_dir):
            os.makedirs(data_list_dir)
            print(f"Created directory: {data_list_dir}")

        all_image_paths = []
        all_label_paths = []

        # Traverse all root directories to collect image and label paths
        for root_dir in WildScenesDataset.root_dirs:
            image_base = os.path.join(root_dir, 'image')
            label_base = os.path.join(root_dir, 'indexLabel')

            if not os.path.exists(image_base) or not os.path.exists(label_base):
                print(f"Error: Image or label directory does not exist in {root_dir}")
                continue

            for image in os.listdir(image_base):
                image_origin = os.path.join(image_base, image)
                image_label = os.path.join(label_base, image)

                if not (os.path.isfile(image_label) and os.path.exists(image_label)):
                    print(f"Warning: Skipping invalid file pair {image_origin}, {image_label}")
                    continue

                all_image_paths.append(image_origin)
                all_label_paths.append(image_label)

        # Create DataFrame with image and label paths
        df = pd.DataFrame(
            data={'image': all_image_paths, 'label': all_label_paths}
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

    # When testing semantic segmentation, use
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

if __name__=='__main__':
    # Example usage
    root_dirs = [
        os.path.join('..', 'data', 'WildScenes', 'WildScenes2d', 'V-01'),
        os.path.join('..', 'data', 'WildScenes', 'WildScenes2d', 'V-02'),
        os.path.join('..', 'data', 'WildScenes', 'WildScenes2d', 'V-03')
    ]

    for root_dir in root_dirs:
        WildScenesDataset.image_file_base = os.path.join(root_dir, 'image')
        WildScenesDataset.label_file_base = os.path.join(root_dir, 'indexLabel')
        WildScenesDataset.make_data_list()

    # Test label mapping for the first image in labelIndex
    # test_label_path = os.path.join(WildScenesDataset.label_file_base,
    #                             '1623370408-092005506.png')  # Replace with an actual image name
    # original_label, mapped_label = WildScenesDataset.test_label_mapping(test_label_path)

    # print(f"Original label array (shape: {original_label.shape}):")
    # print(original_label)
    # print(f"\nMapped trainId array (shape: {mapped_label.shape}):")
    # print(mapped_label)

    # # Optional: print unique values in each array
    # print("\nUnique values in original label array:", np.unique(original_label))
    # print("Unique values in mapped trainId array:", np.unique(mapped_label))
