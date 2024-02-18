from datasets import load_metric

cer_metric = load_metric("cer")
def compute_cer(pred_ids, label_ids, processor):
    """
    Compute the Character Error Rate (CER) between predicted and label sequences.

    Args:
        pred_ids (List[List[int]]): List of predicted token IDs.
        label_ids (List[List[int]]): List of label token IDs.
        processor: The processor object used for decoding.

    Returns:
        float: The computed Character Error Rate (CER).
    """
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer