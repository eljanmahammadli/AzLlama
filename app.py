import gradio as gr
import torch
from transformers import pipeline

model_name = "eljanmahammadli/AzLlama-152M-Alpaca"
model = pipeline("text-generation", model=model_name, torch_dtype=torch.float16)
logo_path = "/Users/eljan/Documents/azGPT/DALL·E 2024-04-08 14.10.01 - A realistic yet adorably cute and sweet llama standing in front of Baku's iconic Flame Towers, now with previously overlooked pixels from the removed .webp"


def get_prompt(question):
    base_instruction = "Aşağıda tapşırığı təsvir edən təlimat və əlavə kontekst təmin edən giriş verilmiştir. Sorğunu uyğun şəkildə tamamlayan cavab yazın."
    prompt = f"""{base_instruction}

### Təlimat:
{question}

### Cavab:
"""
    return prompt


def get_answer(llm_output):
    return llm_output.split("### Cavab:")[1].strip()


def answer_question(history, temperature, top_p, repetition_penalty, top_k, question):
    model_params = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "top_k": top_k,
        "max_length": 512,  # Adjust based on your needs
        "do_sample": True,
    }
    prompt = get_prompt(question)
    llm_output = model(prompt, **model_params)[0]
    answer = get_answer(llm_output["generated_text"])
    divider = "\n\n" if history else ""
    print(answer)
    new_history = history + divider + f"User: {question}\nAssistant: {answer}\n"
    return new_history, ""  # Return updated history and clear the question input


# Define the send_action function before it's referenced
def send_action(_=None):
    send_button.click()


with gr.Blocks() as app:
    gr.Markdown("# AzLlama-150M Chatbot\n\n")

    with gr.Row():
        with gr.Column(scale=0.2, min_width=200):
            gr.Markdown("### Model Logo")
            gr.Image(
                value=logo_path,
            )
            # write info about the model
            gr.Markdown(
                "### Model Info\n"
                "This model is a 150M paramater LLaMA2 model trained from scratch on Azerbaijani text. It can be used to generate text based on the given prompt. "
            )
        with gr.Column(scale=0.6):
            gr.Markdown("### Chat with the Assistant")
            history = gr.Textbox(
                label="Chat History", value="", lines=20, interactive=False
            )
            question = gr.Textbox(
                label="Your question",
                placeholder="Type your question and press enter",
            )
            send_button = gr.Button("Send")
        with gr.Column(scale=0.2, min_width=200):
            gr.Markdown("### Model Settings")
            temperature = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, label="Temperature"
            )
            gr.Markdown(
                "Controls the randomness of predictions. Lower values make the model more deterministic."
            )
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top P")
            gr.Markdown(
                "Nucleus sampling. Lower values focus on more likely predictions."
            )
            repetition_penalty = gr.Slider(
                minimum=1.0, maximum=2.0, value=1.2, label="Repetition Penalty"
            )
            gr.Markdown(
                "Penalizes repeated words. Higher values discourage repetition."
            )
            top_k = gr.Slider(minimum=0, maximum=100, value=50, label="Top K")
            gr.Markdown("Keeps only the top k predictions. Set to 0 for no limit.")

    question.submit(send_action)

    send_button.click(
        fn=answer_question,
        inputs=[history, temperature, top_p, repetition_penalty, top_k, question],
        outputs=[history, question],
    )

app.launch()
