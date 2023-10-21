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

st.title("Text Translation App")

input_text = st.text_input("Enter the text to be translated:")

if input_text:
    translated_text = translate_text(input_text, 'en', tokenizer, model)
    st.write(f"Translated text: {translated_text}")
