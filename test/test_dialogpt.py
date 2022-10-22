from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
#
# def load_tokenizer_and_model(model="microsoft/DialoGPT-medium"):
#     """
#       Load tokenizer and model instance for some specific DialoGPT model.
#     """
#     # Initialize tokenizer and model
#     print("Loading model...")
#     tokenizer = AutoTokenizer.from_pretrained(model)
#     model = AutoModelForCausalLM.from_pretrained(model)
#
#     # Return tokenizer and model
#     return tokenizer, model
#
#
# def generate_response(tokenizer, model, chat_round, chat_history_ids):
#     """
#       Generate a response to some user input.
#     """
#     # Encode user input and End-of-String (EOS) token
#     new_input_ids = tokenizer.encode(input(">> You:") + tokenizer.eos_token, return_tensors='pt')
#
#     # Append tokens to chat history
#     bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_round > 0 else new_input_ids
#
#     # Generate response given maximum chat length history of 1250 tokens
#     chat_history_ids = model.generate(bot_input_ids, max_length=1250, pad_token_id=tokenizer.eos_token_id)
#
#     # Print response
#     print("DialoGPT: {}".format(
#         tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
#
#     # Return the chat history ids
#     return chat_history_ids
#
#
# def chat_for_n_rounds(n=5):
#     """
#     Chat with chatbot for n rounds (n = 5 by default)
#     """
#
#     # Initialize tokenizer and model
#     tokenizer, model = load_tokenizer_and_model()
#
#     # Initialize history variable
#     chat_history_ids = None
#
#     # Chat for n rounds
#     for chat_round in range(n):
#         chat_history_ids = generate_response(tokenizer, model, chat_round, chat_history_ids)
#
#
# if __name__ == '__main__':
#     chat_for_n_rounds(5)