# Fine-Tuning TrOCR for Handwritten Text Recognition

## Introduction
This project focuses on fine-tuning the TrOCR (Transformer-based Optical Character Recognition) model to recognize handwritten text, specifically targeting documents filled out by hand. Given the challenging nature of the task, with datasets comprising images of varying handwriting quality—some of which may be barely legible—this endeavor aims to enhance the model's ability to decipher even the poorest handwriting. For future improvements, employing cyclical GANs for pre-processing images to enhance their quality before recognition is proposed.


## Installation
To set up this project, follow these steps:

- Clone the repository:
```bash
git clone https://github.com/yalsaffar/Fine-tuned-TrOCR
cd Fine-tuned-TrOCR
pip install -r requirements.txt
```
## Usage
This project can be used in two ways: through a Jupyter notebook (`trocer.ipynb`) for an interactive approach, or by running a Python script (`train_trocr.py`) for a more automated process.

### Jupyter Notebook
- Navigate to the `trocer.ipynb` file and follow the instructions within to train and evaluate the model.

### Command Line
- To train the model using the command line, navigate to the project directory and run:
```bash
python train_trocr.py --paths_data_path <path_to_data> --trocr_path <path_to_model> --path_to_save <path_to_save_model> [additional arguments]
```
## Model Details
This project utilizes the "TrOCR-small" model for the task of Optical Character Recognition (OCR) on handwritten text images. While this smaller variant of TrOCR is efficient and effective for the given dataset, exploring larger versions of the TrOCR model could yield better accuracy and performance. Larger models, such as "TrOCR-base" or "TrOCR-large", have more parameters and thus a greater capacity to learn from complex data, which might be beneficial given the challenging nature of the handwritten texts encountered in this project.
## Data Limitations & Advantages
### Limitations:
- The initial dataset comprises cropped images of handwriting alongside their corresponding labels, sourced from a CSV file. A significant challenge is the poor quality of handwriting in many of these images, with some texts being nearly unreadable.
- The variability in handwriting styles and the presence of potentially unreadable texts introduce complexity in training an effective OCR model.

### Advantages:
- Despite the challenging nature of the dataset, using TrOCR allows for leveraging advanced transformer-based techniques to recognize even poorly written texts.
- The project framework is designed to handle images of varying quality, making it adaptable to real-world scenarios where OCR systems often face similar challenges.
## Data Structure
For successful execution of this project, the dataset should be structured as follows:
- A CSV file containing two columns: one for the labels (the transcribed text) and another for the file paths to the corresponding images.
- The images should ideally be cropped to focus on the handwritten text and normalized to the same size to ensure uniformity during model training.
- Example CSV structure:
```
text,img_path
"path/to/image_001.png","Hello, world!"
"path/to/image_002.png","Goodbye!"
```

## Future Work
For future enhancements of this project, considering the application of cyclical GANs (Generative Adversarial Networks) is recommended. These networks can be employed to pre-process the images to significantly improve their quality. This approach would involve training the cyclical GANs on a high-quality dataset before applying them to enhance the handwritten images used in this project. This preprocessing step could potentially lead to better OCR results by making the handwriting more legible for the TrOCR model.


## Configuration
Configuration options include setting the test, validation, and training sizes, model parameters, and learning rates among others, as detailed in the train_trocr py script.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## References

- Original [Paper](https://arxiv.org/abs/2109.10282).
- [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) Model used
