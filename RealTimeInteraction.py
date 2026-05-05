import pandas as pd
import numpy as np
import speech_recognition as sr
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sounddevice as sd
from scipy.io.wavfile import write
from pynput import keyboard
import threading
import librosa
import pickle
import cv2
import ollama
import time

print("Begin Initialization.")

# Load keras models and other required resources
model_audio = load_model("resources/model_audio.keras")
print("Loaded audio model.")
model_text = load_model("resources/model_text.keras")
print("Loaded text model.")
with open('resources/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Loaded tokenizer.")
model_image = load_model("resources/model_image.keras")
print("Loaded image model.")
au_proportions_df = pd.read_csv('resources/au_proportions.csv')
au_proportions = [au_proportions_df[col] for col in au_proportions_df]
labels = {'happy': 0, 'sad': 1, 'angry': 2, 'fear': 3, 'disgust': 4, 'surprise': 5}
model_AU = load_model("resources/model_AU.keras")
print("Loaded AU generator model.")

r = sr.Recognizer()
print("Loaded speech recognizer.")

def preprocess_text(text):
    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])
    
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=250)
    
    return padded_sequence

# Need to make a dummy prediction to initialize cuDNN at the start (If cuDNN is used). Otherwise first prediciton will be slower
model_text(preprocess_text("This is a test sentence.")) 

print("Completed Dummy Prediction.")

# Set the sample rate and data type for the recording
sample_rate = 44100
dtype = np.int16

# Set global variables and flags
recording = None
is_recording = False
is_processing = False
frames = []

# Start 
video = cv2.VideoCapture(0)

# Get the dimensions of the frame
ret, frame = video.read()
height, width, _ = frame.shape

print("Loaded video capture.")

emotion_conv = {
    1 : "happy",
    2 : "sad",
    3 : "angry",
    4 : "fear",
    5 : "disgust",
    6 : "surprise"
}

def LLM_FreshMem():
    return [{"role": "system", "content": "The user will speak to you. " 
                                    + "You will be informed how they feel with (emotion). "
                                    + "Respond or give advice according to their feeling and prompt."},
        ]
msgs = LLM_FreshMem()

print("Created fresh LLM memory.")

print("Completed Initialization.")

# Takes an emotion and text to feed into the LLM (Ensure ollama is running and the model llama2ft is present)
def LLM_Response(emo,text):
    # msg appends to keep memory
    prompt = "("+emo+") "+text
    # append user's prompts
    msgs.append({"role": "user", "content": prompt})
    output = ollama.chat(model="llama2ft",messages=msgs)
    # print AI's message
    print(output['message']['content'])
    # append AI's outputs
    msgs.append({"role": "assistant","content":output['message']['content']})
    resTime_sec = int(int(output['total_duration']) / 1000000000)
    resTime_min = int(resTime_sec / 60)
    resTime_sec = resTime_sec-(resTime_min*60)
    #print("Response time: %s minutes %s seconds" % (resTime_min,resTime_sec))
    # print("Tokens per second: "+ str(int(output['eval_count']) / int(output['eval_duration'])))

# Extract the MFCCs from the audio file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccsscaled

# Combines the image predictions from multiple frames together and averages them
def image_combine(pred_arr):
    image_num = pred_arr.shape[0]
    summed = tf.reduce_sum(pred_arr, axis=0) / image_num
    return tf.expand_dims(summed, axis=0)

# Fusion weighting for the three model outputs
def fusion_mechanism(arr_audio, arr_text, arr_image):
    weighted_sum = (arr_audio * 0.2) + (arr_text * 0.6) + (arr_image * 0.2)
    return weighted_sum

# Returns the AU intensities given an emotion
def get_AU(emotion):
    au_proportion = au_proportions[labels[emotion]]
    au_series = au_proportion.apply(lambda p: np.random.choice([0, 1], p=[1-p, p]))

    emo_cat = np.zeros(6)
    emo_cat[labels[emotion]] = 1

    au_array = np.array(au_series)

    x = np.concatenate((emo_cat, au_array)).reshape(1,24)

    return model_AU(x)

# Begin recording (Spacebar is held)
def start_recording():
    global recording
    global is_recording
    global frames
    global out
    is_recording = True
    frames = []
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))
    recording = sd.InputStream(samplerate=sample_rate, channels=2, dtype=dtype)
    recording.start()
    while is_recording:
        data, overflowed = recording.read(1024)  # read in chunks of 1024 samples
        frames.append(data)
        # Capture video frame
        ret, frame = video.read()
        if ret:
            out.write(frame)

# End recording (Spacebar is released)
def stop_recording():
    start_time = time.time()
    global recording
    global is_recording
    global is_processing
    global frames
    global out
    is_recording = False
    recording.stop()
    recording.close()
    audio_data = np.concatenate(frames, axis=0)  # concatenate all chunks into one array
    # Save the recording to a .wav file
    write('output.wav', sample_rate, audio_data)
    #print('Recording saved to output.wav')
    out.release()
    #print("Video saved to output.avi")

    vid = cv2.VideoCapture('output.avi')
    # Get the total number of frames in the video
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the number of frames per second
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    # Calculate the total number of seconds
    total_seconds = total_frames // fps

    # Array to hold the first frame of every second
    frames_every_second = []
    # For each second in the video
    for i in range(total_seconds):
        # Set the video position to the first frame of the current second
        vid.set(cv2.CAP_PROP_POS_FRAMES, i * fps)
        # Read the frame
        ret, frame = vid.read()
        if ret:
            # Crop the frame to 48x48 around the center
            y, x = frame.shape[0:2]
            start_x = x//2-(48//2)
            start_y = y//2-(48//2)
            cropped_frame = frame[start_y:start_y+48, start_x:start_x+48]
            # Add the cropped frame to the array
            frames_every_second.append(cropped_frame)
    
    frames_every_second = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_every_second]
    frames_every_second = [frame / 255.0 for frame in frames_every_second]
    frames_every_second = [np.expand_dims(frame, axis=2) for frame in frames_every_second]
    frames_every_second = np.array(frames_every_second)

    test_feature = []
    test_feature.append(extract_features("output.wav"))
    test_feature = np.array(test_feature)
    test_feature = np.expand_dims(test_feature, axis=2)

    with sr.AudioFile("output.wav") as source:
        audio_text = r.listen(source)
    
    text = ""

    try:
        # using google speech recognition (may not be the greatest at detecting speech)
        text = r.recognize_google(audio_text)
        print("You said:", text)

        pred_arr_audio = model_audio(test_feature)
        pred_audio = np.argmax(pred_arr_audio, axis=1)[0] + 1
        processed_text = preprocess_text(text)
        pred_arr_text = model_text(processed_text)
        pred_text = np.argmax(pred_arr_text, axis=1)[0] + 1
        pred_arr_image = image_combine(model_image(frames_every_second))
        pred_image = np.argmax(pred_arr_image, axis=1)[0] + 1
        pred_arr_combined = fusion_mechanism(pred_arr_audio, pred_arr_text, pred_arr_image)
        pred_combined = np.argmax(pred_arr_combined, axis=1)[0] + 1

        print(f"Output (Array, Emotion):")
        print(f" - Audio: {pred_arr_audio} - {emotion_conv[pred_audio]} ({pred_audio})")
        print(f" - Text: {pred_arr_text} - {emotion_conv[pred_text]} ({pred_text})")
        print(f" - Image: {pred_arr_image} - {emotion_conv[pred_image]} ({pred_image})")
        print(f" - Combined: {pred_arr_combined} - {emotion_conv[pred_combined]} ({pred_combined})")

        print(f"AU Intensities: {get_AU(emotion_conv[pred_combined])}")

        # Ollama being fed voice text + emotion
        LLM_Response(emotion_conv[pred_combined], text)
        end_time = time.time()
        #print(f"Response Time: {end_time - start_time} seconds")
    except:
        print('Sorry. Please try again.')

    is_processing = False

# Callback functions to be called when a key is pressed or released
def on_press(key):
    global is_processing
    if key == keyboard.Key.space and not is_recording and not is_processing:
        print('Recording...')
        # Start recording
        threading.Thread(target=start_recording).start()
    if key == keyboard.Key.backspace and not is_recording and not is_processing:
        # print("before ",np.array(msgs))
        msgs = LLM_FreshMem()
        # print("after ",np.array(msgs))
        print("Memory cleared, new conversation started.")

def on_release(key):   
    global is_processing
    if key == keyboard.Key.space and is_recording:
        print('Stopped recording')
        # Stop recording
        is_processing = True
        threading.Thread(target=stop_recording).start()

# Create a listener for keyboard events
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    print("Hold 'Spacebar' to talk | Press 'Backspace' to clear memory")
    listener.join()