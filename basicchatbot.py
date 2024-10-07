# Import necessary libraries
import tkinter as tk
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import spacy
import string
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy's pre-trained language model (ensure that the model is installed beforehand)
nlp = spacy.load("en_core_web_md")

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


# Define the basic knowledge base for predefined responses
knowledge_base = {
    "hello": "Hi there! How can I assist you today?",
    "how are you": "I'm just a bot, but I'm doing well! How about you?",
    "what is your name": "I’m ChatGPT, your virtual assistant.",
    "tell me a joke": "Why don’t scientists trust atoms? Because they make up everything!",
    "what can you do": "I can chat with you, help with basic tasks, or answer some questions.",
    "exit": "Goodbye! Have a nice day!"
}

# Context dictionary to store conversational context
context = {}

# Preprocessing: Tokenization, Stemming, and Cleaning
def preprocess(user_input):
    tokens = word_tokenize(user_input.lower())  # Convert to lowercase and tokenize
    stemmer = PorterStemmer()  # Initialize the stemmer
    stop_words = set(stopwords.words('english'))  # Load stopwords
    cleaned_tokens = [word for word in tokens if word not in string.punctuation]  # Remove punctuation

    # Stem each word and remove stopwords
    processed_words = [stemmer.stem(word) for word in cleaned_tokens if word not in stop_words]
    return processed_words

# Function to compute similarity between user input and predefined questions using spaCy
def get_most_similar_response(user_input):
    user_input_doc = nlp(user_input.lower())  # Convert user input to a spaCy document
    best_match = None
    highest_similarity = 0

    for question, answer in knowledge_base.items():
        question_doc = nlp(question)  # Convert each knowledge base question to a spaCy document
        similarity = user_input_doc.similarity(question_doc)  # Calculate similarity

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = answer

    return best_match if highest_similarity > 0.6 else None  # Return None if no good match

# Get contextual response based on conversation history
def get_contextual_response(user_input):
    global context

    # If user asks for the chatbot's name, we remember it
    if "your name" in user_input.lower():
        context["name"] = "ChatGPT"
        return "I'm ChatGPT. What's your name?"

    # If the user tells their name, store it
    if "my name is" in user_input.lower():
        user_name = user_input.split("my name is")[-1].strip()
        context["user_name"] = user_name
        return f"Nice to meet you, {user_name}!"

    # Use the context for personalized conversation
    if "how are you" in user_input.lower():
        if "user_name" in context:
            return f"I'm doing well, {context['user_name']}! How are you?"
        else:
            return "I'm doing well! How about you?"

    # Default to spaCy-based similarity matching
    return None




# Function to generate a response using GPT-2
def generate_gpt2_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')  # Convert text to input tokens
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)  # Generate response
    response = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True)
 # Decode output
 
    return response


# Main chatbot loop
# Tkinter GUI
def run_chatbot_gui():
    # Initialize main window
    window = tk.Tk()
    window.title("AI Chatbot")
    window.geometry("400x500")

    # Add scrollbar
    scrollbar = tk.Scrollbar(window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Text box to display chat
    chat_display = tk.Text(window, height=40, width=200, yscrollcommand=scrollbar.set)
    chat_display.pack(pady=10)
    scrollbar.config(command=chat_display.yview)

    # Entry box for user input
    user_input = tk.Entry(window, width=70)
    user_input.pack(pady=10)

    # Function to handle input
    def process_input(user_query):
        chat_display.insert(tk.END, "You: " + user_query + "\n")

        # Get response based on context or fallback to GPT-2
        response = get_contextual_response(user_query) or generate_gpt2_response(user_query)

        chat_display.insert(tk.END, "Chatbot: " + response + "\n")
        user_input.delete(0, tk.END)
        


    # Bind Enter key to send message
    def on_enter_key(event):
        process_input(user_input.get())
        
        
    # Bind the Enter key
    user_input.bind("<Return>", on_enter_key)    
    # Send button
    send_button = tk.Button(window, text="Send", command=lambda: process_input(user_input.get()))
    send_button.pack(pady=10)

    # Run the GUI
    window.mainloop()

# Run the chatbot
if __name__ == "__main__":
    run_chatbot_gui()
