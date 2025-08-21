# Anime-recommendation
A fine-tuned embedding model for recommending anime from user query.\
This project is heavily inspired from [manga-recommendation](https://github.com/jjentaa/manga-recommendation) by [@jjentaa](https://github.com/jjentaa)

### You can try the web application for the model [here](https://anime-recommendation-ebx77y4id4svvfdamd7uq3.streamlit.app/)
## Fine-tune model
1. Using `mistralai/Mistral-7B-Instruct-v0.2` for generated query and use it for fine tuning pre-trained model
2. I use `paraphrase-multilingual-mpnet-base-v2` as a base model + LoRA adapter, fine-tuning with anime synopsis data + query\
   \
[KiruruP/anime-recommendation-multilingual-mpnet-base-v2-peft](https://huggingface.co/KiruruP/anime-recommendation-multilingual-mpnet-base-v2-peft): Multilingual Fine-tuned model

## Want to run on local? Following this guideline!!
1. Clone this repository
2. Install the necessary library follwing the list below or you can install in terminal through command `pip install -r requirements.txt`
```
sentence_transformers == 5.1.0
numpy == 2.1.3
torch == 2.7.1
pandas == 2.2.3
streamlit == 1.48.1
peft
```
3. run this command in terminal, it will open in your browser and enjoy!! :)
```
 streamlit run app.py
```
