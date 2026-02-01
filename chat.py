import random
import json
import torch
import os
from duckduckgo_search import DDGS
from nltk_utils import bag_of_words, tokenize
from train import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. LOAD LOCAL BRAIN
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
model.load_state_dict(data["model_state"])
model.eval()

last_topic = "" # Global variable to track what we talked about

def search_and_remember(query):
    global last_topic
    
    # If query is vague like "what is that", use the last topic we found
    search_query = query
    if any(word in query.lower() for word in ["that", "it", "this"]) and last_topic:
        search_query = f"{query} {last_topic}"
    
    print(f"(NanduBot is searching the web for: {search_query}...)")
    try:
        with DDGS() as ddgs:
            # max_results=2 gives a better summary than just 1
            results = [r for r in ddgs.text(search_query, max_results=2)]
            if results:
                # Combine snippets and save to memory
                answer = " ".join([r['body'] for r in results])
                
                # Update our topic tracker
                if "fortnite" in search_query.lower(): last_topic = "Fortnite"
                
                with open("brain_memory.txt", "a") as f:
                    f.write(f"Question: {search_query} | Answer: {answer[:200]}\n")
                return f"I found this: {answer[:300]}..."
    except:
        return "I'm having trouble connecting to my search engine right now."
    
    return "I couldn't find a clear answer. Could you be more specific?"

# 3. CHAT LOOP
bot_name = "NanduBot"
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit": break

    # Check Local Memory File First
    found_in_memory = False
    if os.path.exists("brain_memory.txt"):
        with open("brain_memory.txt", "r") as f:
            for line in f:
                if sentence.lower() in line.lower():
                    print(f"{bot_name} (from memory): {line.split('| Answer: ')[1].strip()}")
                    found_in_memory = True
                    break
    if found_in_memory: continue

    # Check Neural Network
    X = bag_of_words(tokenize(sentence), data["all_words"])
    X = torch.from_numpy(X.reshape(1, X.shape[0])).to(device)
    output = model(X)
    prob = torch.softmax(output, dim=1)[0][torch.max(output, dim=1)[1].item()]

    if prob.item() > 0.75:
        tag = data["tags"][torch.max(output, dim=1)[1].item()]
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        # Web Search Fallback
        print(f"{bot_name}: {search_and_remember(sentence)}")