
import streamlit as st
from transformers import M2M100ForConditionalGeneration
from transformers import AutoTokenizer

def translate_text(input_text, target_language, tokenizer, model):
    tokenizer.tgt_lang = target_language
    encoded_text = tokenizer(input_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text)
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text

model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = AutoTokenizer.from_pretrained("alirezamsh/small100")

languages = [
    "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb",
    "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "fy",
    "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig",
    "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg",
    "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne",
    "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru",
    "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su",
    "sv", "sw", "ta", "th", "tl", "tn",
    "tr", "uk", "ur", "uz",
    "vi",
    "wo",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu"
]

st.title("Text Translation App")

input_text = st.text_input("Enter the text to be translated:")

source_language = st.selectbox("Select the source language:", languages)
target_language = st.selectbox("Select the target language:", languages)

if input_text:
    translated_text = translate_text(input_text, target_language, tokenizer, model)
    st.write(f"Translated text: {translated_text}")
