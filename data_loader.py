import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader

def load_data_paths(path):
    """
    Load data paths from a CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data paths.
    """
    df = pd.read_csv(path)
    return df

def split_data(df, random_seed=42, test_size=0.2, val_size=0.2):
    """
    Split the data into train, test, and validation sets.

    Args:
        df (pandas.DataFrame): The input data.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        test_size (float, optional): The proportion of the data to include in the test set. Defaults to 0.2.
        val_size (float, optional): The proportion of the data to include in the validation set. Defaults to 0.2.

    Returns:
        tuple: A tuple containing train, test, and validation sets.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_seed)
    train, val = train_test_split(train, test_size=val_size, random_state=random_seed)
    return train, test, val

class HandWrittenDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        """
        Custom dataset for handling handwritten data.

        Args:
            root_dir (str): The root directory of the dataset.
            df (pandas.DataFrame): The input data.
            processor: The image processor.
            max_target_length (int, optional): The maximum length of the target text. Defaults to 128.
        """
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary containing the image pixel values and labels.
        """
        # get file name + text 
        file_name = self.df['img_path'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def batch_data_loader(train_dataset, eval_dataset, batch_size=24):
    """
    Create batch data loaders for training and evaluation.

    Args:
        train_dataset (HandWrittenDataset): The training dataset.
        eval_dataset (HandWrittenDataset): The evaluation dataset.
        batch_size (int, optional): The batch size. Defaults to 24.

    Returns:
        tuple: A tuple containing the training and evaluation data loaders.
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    return train_dataloader, eval_dataloader