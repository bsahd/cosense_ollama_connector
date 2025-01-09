# Cosense Ollama Connector

The Cosense Ollama Connector is a simple script for connecting Cosense and Ollama.

The script is designed so that developers can easily grasp the big picture and customize it to their own needs. Also, the purpose of the project is to show a simple implementation, not to satisfy a wide variety of needs. I encourage everyone to understand the source code and customize it to their own needs.

## For Japanese reader
Visit https://scrapbox.io/villagepump/bsahd%2Fcosense_ollama_connector


## How to install

Clone the GitHub repository.

Run the following commands to install the required libraries.

```sh
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

install Ollama models.
```sh
ollama pull kun432/cl-nagoya-ruri-large # Japanese-tuned BERT-Based embedding model
ollama pull hf.co/alfredplpl/llm-jp-3-1.8b-instruct-gguf:IQ4_XS # Small size LLM for Japanese
```

## How to use

Make index.

$ python make_index.py

It outputs like below:

```
 % python make_index.py
 100%|█████████████████████████████████████| 872/872 [07:45<00:00, 1 .87it/s] 
 ```

Ask. 

$ python ask.py

It outputs like below:

```
>>>> What is the most important question?
> The most important question is to know ourselves.
```

# License
The Cosense Ollama Connector is distributed under the MIT License. See the LICENSE file for more information.