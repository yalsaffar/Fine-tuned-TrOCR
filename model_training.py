from transformers import TrOCRProcessor, AdamW
import torch
from transformers import VisionEncoderDecoderModel
from tqdm.notebook import tqdm
from eval_metric import compute_cer

def load_processor(path):
    """
    Load the TrOCRProcessor from the given path.

    Args:
        path (str): The path to the pre-trained processor.

    Returns:
        TrOCRProcessor: The loaded TrOCRProcessor.
    """
    processor = TrOCRProcessor.from_pretrained(path)
    return processor

def load_model(path):
    """
    Load the VisionEncoderDecoderModel from the given path.

    Args:
        path (str): The path to the pre-trained model.

    Returns:
        VisionEncoderDecoderModel: The loaded VisionEncoderDecoderModel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionEncoderDecoderModel.from_pretrained(path)
    model.to(device)
    return model

def set_model_params(model, processor, max_length, early_stopping, no_repeat_ngram_size, length_penalty, num_beams):
    """
    Set the parameters of the model.

    Args:
        model (VisionEncoderDecoderModel): The model to set the parameters for.
        processor (TrOCRProcessor): The processor used for tokenization.
        max_length (int): The maximum length of the generated output.
        early_stopping (bool): Whether to stop generation early.
        no_repeat_ngram_size (int): The size of n-grams to avoid repeating.
        length_penalty (float): The length penalty for beam search.
        num_beams (int): The number of beams for beam search.

    Returns:
        VisionEncoderDecoderModel: The model with updated parameters.
    """
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = max_length
    model.config.early_stopping = early_stopping
    model.config.no_repeat_ngram_size = no_repeat_ngram_size
    model.config.length_penalty = length_penalty
    model.config.num_beams = num_beams

    return model

def train_model(model, train_dataloader, eval_dataloader, optimizer, device):
    """
    Train the model.

    Args:
        model (VisionEncoderDecoderModel): The model to train.
        train_dataloader (DataLoader): The dataloader for training data.
        eval_dataloader (DataLoader): The dataloader for evaluation data.
        optimizer (Optimizer): The optimizer for updating model parameters.
        device (str): The device to use for training (e.g., "cuda" or "cpu").

    Returns:
        VisionEncoderDecoderModel: The trained model.
    """
    for epoch in range(3):  # loop over the dataset multiple times
        # train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            # get the inputs
            for k, v in batch.items():
                batch[k] = v.to(device)

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))

        # evaluate
        model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device))
                # compute metrics
                cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                valid_cer += cer

        print("Validation CER:", valid_cer / len(eval_dataloader))

    return model

def save_model(model, path):
    """
    Save the model to the given path.

    Args:
        model (VisionEncoderDecoderModel): The model to save.
        path (str): The path to save the model.

    Returns:
        VisionEncoderDecoderModel: The saved model.
    """
    model.save_pretrained(path)
    return model
