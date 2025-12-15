# LLM: Legal Layman's Mansplaininator
A Retrieval-Augmented AI System for Translating Forensic Testimony & Highlighting Legal Compliance Issues

## What is LLM?
_If you're looking for a definition of Large Language Models, please read [this Medium article.](https://medium.com/data-science-at-microsoft/how-large-language-models-work-91c362f5b78f)_

_LLM: Legal Layman’s Mansplaininator_ is an interactive AI tool designed to help legal professionals and digital forensic practitioners bridge the gap between highly technical forensic testimony and courtroom-ready explanations.

The system provides:

### 1. A Layman Translator
Converts technical terminology into clear, accessible language suitable for judges, juries, and attorneys.

### 2. A Compliance Highlighter
Analyzes testimony or user questions and surfaces potential issues related to admissibility, chain of custody, and procedural compliance based on NIJ, DOJ, and related guidelines.
    
### 3. A Prompt Evaluator
Allows comparison of alternative prompt designs using an LLM-as-a-Judge scoring system.

## System Architecture
The system combines:
- Retrieval-Augmented Generation (RAG)
- Local LLMs served via Ollama
- FAISS vector indexing
- A Streamlit interface for real-time interaction

The user is able to upload PDFs or paste links (.pdf, .txt, .html are currently supported) to the tool. Underneath the hood, these documents are vectorized and stored in separate FAISS instances for each use case: the _Technical-to-Layman Translator_ and the _Compliance Highlighter_. To retrieve relevant documents (or "chunks", in fancier terms), the user will need to input a query in a way that a digital forensics expert would be asked during a witness testimony. 

The performance of an LLM is entirely dependent on its system prompt. With the intention of broadening our search for the perfect (or near-perfect!) system prompt, we have also exposed an Evaluator module. In this, we harness a different LLM to act as a judge to compare multiple system prompt candidates over a preset of 3 questions. The judgment (a score from 1 to 5) is based on the following metrics:
- Technical correctness
- Completeness
- Clarity for legal professionals
- Faithefulness to source context

The LLM produces a summary with suggestions for improvement.

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/g-aditi/legal-laymans-mansplaininator.git
cd legal-laymans-mansplaininator
```

### 2. Creating a Python environment

#### Using Conda
```bash
conda create -n mansplaininator python=3.10.16
conda activate mansplaininator
pip install -r requirements.txt
```

#### Using Mamba on ASU Sol HPC
```bash
module load mamba/latest
mamba create -n mansplaininator python=3.10.16
mamba activate mansplaininator
pip install -r requirements.txt
```

### 3. Installing and Running Ollama
Ollama is a lightweight platform to run LLMs and requires separate installation. It must be running locally for the tool to execute.

#### On non-ASU Sol devices
Follow [download instructions](https://ollama.com/download) for your OS.

#### On ASU Sol

##### Starting an interactive shell session
Please ensure you are on an interactive shell session. Ollama will NOT run on a login node. You can do so either through the UI or through the CLI with this command.
```bash
interactive -t 30 -G 1 -p htc
```

##### Loading the ollama module
```bash
module load ollama/0.3.12
```

##### Starting Ollama serve in the background
```bash
ollama-start
```

#### Next Ollama steps for all modes
The models used in the tool must all be pulled for usage. You can do so using these commands.
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
ollama pull mistral
```

### 4. Running the application
Launch Streamlit.
```bash
streamlit run app.py
```
Your browser should open to `http://localhost:8501`.

### 5. Configuration
All the LLM settings are defined in `models.py`.
```bash
OLLAMA_URL = f"http://{socket.gethostname()}:11434"
LLM_MODEL = "llama3.2"
EMBED_MODEL = "mxbai-embed-large"
```
You can change these to any local Ollama model. Please make sure you are using `ollama pull` to pull the model you intend to use.

### 6. Data indexing

FAISS indices are stored in:
* `data/faiss_translator/`
* `data/faiss_compliance/`

Documents are stored in:
* `data/docs_translator/` 
* `data/docs_compliance/`

Indices are automatically created when the `Index Documents` button is clicked on the tool.

## Contributors
* Ariadne Dimarogona
* Easton Kelso
* Aditi Ganapathi

We made this project for our Digital Forensics class in Fall 2025. Shout out to Dr. Ahn and the entire class for all the awesome feedback!
