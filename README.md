# Authors

Keanu Chen
Richard Zhang
Marcus Yau

# Overview

Our program looks to combine multimodal user input such as speech, audio and text (extracted from speech) in order to infer an emotion from a user based on their voice and expressions.

This inferred emotion will then be passed to a Large Language Model (LLM) with text in order to generate a response.

# Requirements

Below are the requirements for the program.
- Python 3.9.1 (Tested and compiled on this version)
- Ollama (https://ollama.com/download)
- Check requirements.txt for specific packages
- Camera Device and Microphone Device

# File Structure
```
.
├── Dataset
│   ├── dataset.csv         # Audio and Video dataset with annotated groundtruth
│   ├── dataset_text.csv    # Text dataset with annotated groundtruth
│   ├── convert.bat         # Batch script to separate audio from video
│   └── *.mp4 / *.wav       # Collection of audio/video samples for dataset
├── resources               
│   ├── au_proportions.csv  # For generating AU intensities
│   ├── fer2013_complete    # FER2013 dataset containing AU values for each image
│   ├── model_AU.keras      # AU intensity model
│   ├── model_audio.keras   # Audio Feature model
│   ├── model_image.keras   # Facial Expression model
│   ├── model_text.keras    # Text Sentiment model
│   └── tokenizer.pickle    # Tokenizer for text sentiment
├── AudioProcessing.ipynb   # Training notebook for Audio Features
├── TextProcessing.ipynb    # Training notebook for Text Sentiment
├── FacialProcessing.ipynb  # Training notebook for Facial Expression Analysis
├── MetricGathering.ipynb   # Notebook for gathering results for validation and test
├── LLM-FineTine.ipynb      # Notebook for fine-tuning the LLM (Requires additional packages listed in notebook header)
├── RealTimeInteraction.py  # Main Program
└── ...
```

# Installation

There are a couple of things that need to be installed before the program works.

## Packages
Please run the following command to install Python dependencies through the requirements file.
```
pip install -r requirements.txt
```

## Ollamas
The LLM we are using is based on Ollama and requires the program to run in order to connect to a localhost instance.

1. Go to the link to install Ollama listed in the Requirements section.
2. Once Ollama is installed, run the program to start the instance.
3. Download our fine-tuned llama2 model (3.4G) (https://1drv.ms/u/s!AiTvNZSAilOngYYoRN1kH-WsT-hifA?e=hB5Nd4)
4. Extract the blobs and manifests folders
5. Copy the blobs content into models/blobs and manifests into manifests into models/manifests Ollama folder.
    - This folder by default is located at C:/Users/\<User>/.ollama
6. Verify that the **llama2ft** model was correctly installed by using the command:
```
ollama list
```

## Datasets
If you want to try testing the model training notebooks (AudioProcessing.ipynb, FacialProcessing.ipynb, TextProcessing.ipynb), there are a few datasets that need to be downloaded as a few of them are too large to include in the submission.

### AudioProcessing.ipynb
- Dataset Link: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- You would need to change the path in audio_paths to the location of the dataset.

### FacialProcessing.ipynb
- Dataset Link: https://www.kaggle.com/datasets/msambare/fer2013
- You need to change the two folder variables in the notebook to direct to the train and test folders of the dataset.
- The Action Unit extracted dataset is already included in resources. We just need the images.

### TextProcessing.ipynb
- Dataset Link: https://github.com/google-research/google-research/tree/master/goemotions/data/full_dataset
- This dataset does not have a direct download and above is a link to the instructions to download through wget.
- The dataset is divided into three different files and so you need to alter the datapaths list to point to all three files once downloaded.

### MetricGathering.ipynb
- Test datasets we created for an interaction model are stored in the subfolder Dataset.
- Contains audio, video, batch script to get audio from video and two csv files.
    - dataset.csv: contains each of the 21 file names with an annotated groundtruth
    - dataset_text.csv: contains 20 phrases with an annotated groundtruth

### LLM_FineTuning.ipynb
Dataset Link: https://huggingface.co/datasets/YauMarcus/postconversion
- Created small dataset to fine tune the LLM model
- Required emotion + text as input to LLM

# Running the Program
The program code is in the file RealTimeInteraction.py. You should be able to just run the code.

Once the program starts, it initializes by loading the models and required resources as well as doing a dummy prediction. This is done so that it can initialize CUDA Deep Neutal Network (cuDNN) (if used) ahead of time rather than on first prediciton.

Once ready you should see the following message:
```
Hold 'Spacebar' to talk | Press 'Backspace' to clear memory
```

At this point you can do one of two things:
1. Hold down spacebar to talk to the system and release when you are done talking
    - Try holding the spacebar down an extra 1-2 seconds after you are speaking to achieve better consistency in results
    - Built in speech recognizer sometimes is not able to recognize your speech
2. Press Backspace to clear the LLM's memory to start a new conversation

# Reflection
Between our proposal and current application, we would say that it came pretty close to what we were proposing initially. We essentially wanted to create a system to be able to take multimodal input from a user, process the data and output a response. The differences were that we decided to use a large language model in the end instead of a decision tree, using action units as a way to analyze facial expression was scrapped and we used mel frequency ceptral coefficients (MFCCs) to analyze the vocal patterns instead of piutch, speed and dynamics. We also did not get a chance to adapt our system to using test to speech synthesis, but that was originally out of scope and additional work if time allowed. Most of the design choices that were made to initial plan were due to limited computing resources and data which required us to adapt to what we had to work with. Overall, we think that this system works decently well as an introduction and exploration into real-time affective computing.

