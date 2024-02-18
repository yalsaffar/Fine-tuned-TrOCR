import argparse
from data_loader import *
from model_training import *


def train_trocer(paths_data_path, trocr_path, path_to_save, test_size=0.2, val_size=0.2, max_length=64, early_stopping=True, no_repeat_ngram_size=3, length_penalty=2.0, num_beams=4, learning_rate=5e-5, batch_size=24):
    """
    Train the TrOCR model.

    Args:
        paths_data_path (str): The file path to the data paths file.
        trocr_path (str): The file path to the TrOCR model.
        path_to_save (str): The directory path to save the trained model.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.2.
        max_length (int, optional): The maximum length of the input sequences. Defaults to 64.
        early_stopping (bool, optional): Whether to use early stopping during training. Defaults to True.
        no_repeat_ngram_size (int, optional): The size of the n-gram to avoid repeating in the generated sequences. Defaults to 3.
        length_penalty (float, optional): The length penalty factor for beam search. Defaults to 2.0.
        num_beams (int, optional): The number of beams for beam search. Defaults to 4.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 5e-5.
        batch_size (int, optional): The batch size for training. Defaults to 24.
    """
    df = load_data_paths(paths_data_path)
    train, test, val = split_data(df, random_seed=42, test_size=test_size, val_size=val_size)
    processor = load_processor(trocr_path)

    print("Data loaded and split successfully.")

    train_dataset = HandWrittenDataset(root_dir='.',
                                       df=train,
                                       processor=processor)
    eval_dataset = HandWrittenDataset(root_dir='.',
                                      df=val,
                                      processor=processor)
    test_dataset = HandWrittenDataset(root_dir='.',
                                      df=test,
                                      processor=processor)
    train_dataloader, eval_dataloader = batch_data_loader(train_dataset, eval_dataset, batch_size=batch_size)

    print("Datasets created and dataloaders prepared.")

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))
    print("Number of testing examples:", len(test_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(trocr_path)
    model = set_model_params(model, processor, max_length=max_length, early_stopping=early_stopping, no_repeat_ngram_size=no_repeat_ngram_size, length_penalty=length_penalty, num_beams=num_beams)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("Model loaded and optimizer initialized.")

    model = train_model(model, train_dataloader, eval_dataloader, optimizer, device)

    print("Model training completed.")

    model = save_model(model, path_to_save)

    print("Model saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TrOCR model.")
    parser.add_argument("--paths_data_path", type=str, help="The file path to the paths' data file.")
    parser.add_argument("--trocr_path", type=str, help="The file path to the TrOCR model.")
    parser.add_argument("--path_to_save", type=str, help="The directory path to save the trained model.")
    parser.add_argument("--test_size", type=float, default=0.2, help="The proportion of the dataset to include in the test split.")
    parser.add_argument("--val_size", type=float, default=0.2, help="The proportion of the dataset to include in the validation split.")
    parser.add_argument("--max_length", type=int, default=64, help="The maximum length of the input sequences.")
    parser.add_argument("--early_stopping", type=bool, default=True, help="Whether to use early stopping during training.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="The size of the n-gram to avoid repeating in the generated sequences.")
    parser.add_argument("--length_penalty", type=float, default=2.0, help="The length penalty factor for beam search.")
    parser.add_argument("--num_beams", type=int, default=4, help="The number of beams for beam search.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=24, help="The batch size for training.")

    args = parser.parse_args()

    train_trocer(args.paths_data_path, args.trocr_path, args.path_to_save, args.test_size, args.val_size, args.max_length, args.early_stopping, args.no_repeat_ngram_size, args.length_penalty, args.num_beams, args.learning_rate, args.batch_size)
