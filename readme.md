# NanduBot: A Hybrid AI Chatbot from Scratch

NanduBot is a dual-layered AI assistant built to run on entry-level hardware. It uses a local Neural Network for fast, intent-based responses and a web-search fallback for real-time internet learning.

## 🚀 Features
- **Local Inference:** Runs a PyTorch Neural Network on the NVIDIA GTX 1630.
- **Internet Skills:** Automatically searches the web for answers it hasn't been trained on.
- **Persistent Memory:** Saves web search results to `brain_memory.txt` to "learn" and respond faster next time.
- **Optimized for Budget PCs:** Specifically designed to run smoothly on older CPUs (Intel 3rd Gen).

## 🛠️ Hardware Requirements
- **CPU:** Intel i5-3rd Gen (or equivalent)
- **GPU:** NVIDIA GeForce GTX 1630 (4GB VRAM)
- **RAM:** 12GB+ recommended
- **OS:** Ubuntu / Linux

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd chatbot


Setup Virtual Environment:

``` Bash
python3 -m venv aivenv
source aivenv/bin/activate
Install Dependencies:
```
``` Bash
./aivenv/bin/python3 -m pip install torch numpy nltk ddgs
```
Download NLTK Data:

```Bash
./aivenv/bin/python3 -m nltk.downloader punkt_tab
```

🎮 How to Use
1. Training
If you modify intents.json, you must re-train the local brain:

```Bash
./aivenv/bin/python3 train.py
```
2. Chatting
Start the bot and talk to it in the terminal:

```Bash
./aivenv/bin/python3 chat.py
```
📂 Project Structure
train.py: The script that builds the model using the GTX 1630.

chat.py: The main bot interface with web-search logic.

nltk_utils.py: Text processing (tokenization and stemming).

intents.json: The local knowledge base.

data.pth: The trained model weights.
