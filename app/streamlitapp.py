# Import all of the dependencies
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os 
import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Custom CSS for enhanced aesthetics
st.markdown(
    """
    <style>
        /* Change font family and color */
        body {
            font-family: Arial, sans-serif;
            color: #333;
            background-color: #f0f0f0; /* Light gray background */
        }
        /* Change title font size and color */
        h1 {
            font-size: 36px !important;
            color: #008080 !important; /* Dark cyan title color */
            text-align: center;
        }
        /* Change selectbox font size and color */
        .st-ax {
            font-size: 16px !important;
            color: #333 !important; /* Dark gray selectbox text color */
            background-color: #fff !important; /* White selectbox background */
        }
        /* Change text font size and color */
        .st-cj {
            font-size: 20px !important; /* Larger text size */
            font-weight: bold !important; /* Bold text */
            color: #666 !important; /* Dark gray text color */
            text-align: center;
        }
        /* Adjust column width */
        .st-dd {
            width: 50%;
        }
        /* Box styling */
        .highlight-box {
            padding: 20px;
            background-color: #d3e0ea; /* Light blue */
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('LipLingo') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('data', 's1'))
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        file_path = os.path.join('data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes, format='video/mp4')

    with col2: 
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        
        sample = load_data(tf.convert_to_tensor(f'.\\data\\s1\\{selected_video}'))
        arr = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in [sample[1]]]
        frames_ = []
        for s in arr:
            string_value = s.numpy().decode('utf-8')
            frames_.append(string_value)

        tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        tokens = tokenizer.encode(frames_[0], return_tensors='pt')
        result = model(tokens)
        sentiment = int(torch.argmax(result.logits))+1
        sentiment_ = ""
        if sentiment <= 1:
            t = (result.logits[0][0])
            if t.item() <= 2:
                sentiment_ = "Angry"
            else:
                sentiment_ = "Sad"
        elif sentiment >= 4:
            sentiment_ = "Happy"
        else:
            sentiment_ = "Neutral"
        
        # Highlighted box for extracted text and sentiment
        with st.container():
            st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
            st.markdown("<h2 class='st-cj'>Extracted Text and Sentiment</h2>", unsafe_allow_html=True)
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown(f"<p class='st-cj'>Spoken Sentence: {frames_[0]}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='st-cj'>Sentiment: {sentiment_}</p>", unsafe_allow_html=True)
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)




        