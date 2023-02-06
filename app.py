import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import pickle
from model import EncoderCNN, DecoderRNN, Attention, EncoderDecoder

st.header('Image captioning')

@st.cache
def load_model():
    model = torch.load('models/model_image_captioning.pt', map_location=torch.device('cpu'))
    return model

@st.cache
def load_vocab():
    with open('models/dataset_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab


model = load_model()
st.success('Model is loaded!', icon="✅")
# with st.expander('See model description'):
#     st.(model.encoder)

vocab = load_vocab()
st.success('Vocabulary is loaded!', icon="✅")

def transform_image(img):
    transformer = transforms.Compose([
        transforms.Resize(226),                     
        transforms.RandomCrop(224),                 
        transforms.ToTensor(),                               
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    return transformer


st.write('## Upload your image and find out its caption!')

uploaded_image = st.file_uploader('Upload', type=['png', 'jpg', 'jpeg'])

if not uploaded_image:
    st.write('### I am waiting...')
else:
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(uploaded_image)
        st.write('#### Oringinal image')
        st.image(image)
    with col2:
        transformer = transform_image(uploaded_image)
        transformed_image = transformer(image).view((1, 3, 224, 224))
        model.eval()
        with torch.no_grad():
            features = model.encoder(transformed_image[0:1].to('cpu'))
            caps, alphas = model.decoder.generate_caption(features, vocab=vocab)
            caption = ' '.join(caps[:-2])
            st.write('#### Caption')
            st.write('##### <unk> stands for unknown word')
            st.write(f'# {caption.capitalize()}')