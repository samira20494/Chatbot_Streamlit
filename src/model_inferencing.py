import streamlit as st
import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering
from transformers import AutoTokenizer
import json
from pathlib import Path



@st.cache(allow_output_mutation=True)
def instantiate_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    return tokenizer, model


def load_context_for_inference(path):
    path = Path(path)
    with open(path, 'rb') as f:
        json_data = json.load(f)
        data = json_data["data"]

    context = ""

    for section in range(0, len(data)):
        for parag in range(0, len(data[section]["paragraphs"])):
            context += data[section]["paragraphs"][parag]["context"]

    return context


if __name__ == '__main__':
    st.title('Covid-19 Chatbot')
    tokenizer, model = instantiate_model()

    question = st.text_area('Hi, I am a bot here to answer your covid-19 questions. How may I help you?')

    if question:
        text = load_context_for_inference("COVID-QA.json")

        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="tf", truncation=True)
        input_ids = inputs["input_ids"].numpy()[0]

        output = model(inputs)
        answer_start = tf.argmax(
            output.start_logits, axis=1
        ).numpy()[0]  # Get the most likely beginning of answer with the argmax of the score
        answer_end = (
                tf.argmax(output.end_logits, axis=1) + 1
        ).numpy()[0]  # Get the most likely end of answer with the argmax of the score
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        start = answer.index("SEP") + 4
        answer = answer[start:-5]
        st.markdown(f'Answer is: {answer}')
    # st.markdown('Answer is empty for now!')


