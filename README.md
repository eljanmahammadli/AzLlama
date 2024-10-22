# AzLlama
Documentation soon...
This needs a lot of rafactoring. No time for that yet :(

## Very briefly for now:
- We have prepared one of the largest pre-training corpus of 3 billion tokens of training data in the Azerbaijani language.
- We have around 13 stages of data filtering and curation pipeline customized for the Azerbaijani language.
- We have trained custom SentencePiece BPE tokenizer that is very good at Azerbaijani.
- We have pre-trained 150M LLaMA-based generative (decoder-only transformer) on 3B tokens for 2-3 epochs.
- We have also trained the SFT (Supervised Fine-Tuned) model to give it chatbot-like behavior.
- We have evaluated based on custom metrics.
