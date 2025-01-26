---
title: "NLP Challenge: Authentic Sentence Detection and Synthetic Corruption Generation"
excerpt: "My experience with a challenging NLP task that involved distinguishing English sentences from their corrupted versions and generating new corruptions. <br/><img src='/images/lstm-2.svg'>"
collection: portfolio
---

Source code
------
import streamlit as st
import numpy as np
import numpy.linalg as la
import pickle
import os
import gdown
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import math

def load_glove_embeddings(glove_path="Data/embeddings.pkl"):
    with open(glove_path, "rb") as f:
        embeddings_dict = pickle.load(f, encoding="latin1")

    return embeddings_dict

def get_model_id_gdrive(model_type):
    if model_type == "25d":
        word_index_id = "13qMXs3-oB9C6kfSRMwbAtzda9xuAUtt8"
        embeddings_id = "1-RXcfBvWyE-Av3ZHLcyJVsps0RYRRr_2"
    elif model_type == "50d":
        embeddings_id = "1DBaVpJsitQ1qxtUvV1Kz7ThDc3az16kZ"
        word_index_id = "1rB4ksHyHZ9skes-fJHMa2Z8J1Qa7awQ9"
    elif model_type == "100d":
        word_index_id = "1-oWV0LqG3fmrozRZ7WB1jzeTJHRUI3mq"
        embeddings_id = "1SRHfX130_6Znz7zbdfqboKosz-PfNvNp"
        
    return word_index_id, embeddings_id


def download_glove_embeddings_gdrive(model_type):
    # Get glove embeddings from Google Drive
    word_index_id, embeddings_id = get_model_id_gdrive(model_type)

    # Use gdown to download files from Google Drive
    embeddings_temp = "embeddings_" + str(model_type) + "_temp.npy"
    word_index_temp = "word_index_dict_" + str(model_type) + "_temp.pkl"

    # Download word_index pickle file
    print("Downloading word index dictionary....\n")
    gdown.download(id=word_index_id, output=word_index_temp, quiet=False)

    # Download embeddings numpy file
    print("Downloading embeddings...\n\n")
    gdown.download(id=embeddings_id, output=embeddings_temp, quiet=False)


# @st.cache_data()
def load_glove_embeddings_gdrive(model_type):
    word_index_temp = "word_index_dict_" + str(model_type) + "_temp.pkl"
    embeddings_temp = "embeddings_" + str(model_type) + "_temp.npy"

    # Load word index dictionary
    word_index_dict = pickle.load(open(word_index_temp, "rb"), encoding="latin")

    # Load embeddings numpy array
    embeddings = np.load(embeddings_temp)

    return word_index_dict, embeddings


@st.cache_resource()
def load_sentence_transformer_model(model_name):
    sentenceTransformer = SentenceTransformer(model_name)
    return sentenceTransformer


def get_sentence_transformer_embeddings(sentence, model_name="all-MiniLM-L6-v2"):
    """
    Get sentence transformer embeddings for a sentence
    """
    # 384-dimensional embedding
    # Default model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  

    sentenceTransformer = load_sentence_transformer_model(model_name)

    try:
        return sentenceTransformer.encode(sentence)
    except:
        if model_name == "all-MiniLM-L6-v2":
            return np.zeros(384)
        else:
            return np.zeros(512)


def get_glove_embeddings(word, word_index_dict, embeddings, model_type):
    """
    Get GloVe embedding for a single word
    """
    if word.lower() in word_index_dict:
        return embeddings[word_index_dict[word.lower()]]
    else:
        return np.zeros(int(model_type.split("d")[0]))


def get_category_embeddings(embeddings_metadata):
    """
    Get embeddings for each category
    1. Split categories into words
    2. Get embeddings for each word
    """
    model_name = embeddings_metadata["model_name"]
    st.session_state["cat_embed_" + model_name] = {}
    for category in st.session_state.categories.split(" "):
        if model_name:
            if not category in st.session_state["cat_embed_" + model_name]:
                st.session_state["cat_embed_" + model_name][category] = get_sentence_transformer_embeddings(category, model_name=model_name)
        else:
            if not category in st.session_state["cat_embed_" + model_name]:
                st.session_state["cat_embed_" + model_name][category] = get_sentence_transformer_embeddings(category)


def update_category_embeddings(embeddings_metadata):
    """
    Update embeddings for each category
    """
    get_category_embeddings(embeddings_metadata)


### Plotting utility functions
    
def plot_piechart(sorted_cosine_scores_items):
    sorted_cosine_scores = np.array([
            sorted_cosine_scores_items[index][1]
            for index in range(len(sorted_cosine_scores_items))
        ]
    )
    categories = st.session_state.categories.split(" ")
    categories_sorted = [
        categories[sorted_cosine_scores_items[index][0]]
        for index in range(len(sorted_cosine_scores_items))
    ]
    fig, ax = plt.subplots()
    ax.pie(sorted_cosine_scores, labels=categories_sorted, autopct="%1.1f%%")
    st.pyplot(fig)  # Display figure


def plot_piechart_helper(sorted_cosine_scores_items):
    colors = plt.cm.Pastel1.colors  
    categories = st.session_state.categories.split(" ")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    labels = [categories[i] for i, _ in sorted_cosine_scores_items]
    sizes = [score for _, score in sorted_cosine_scores_items]
    explode = np.zeros(len(labels))
    explode[0] = 0.1
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=lambda p: f'{p:.1f}%',
        startangle=90,
        shadow=True,
        pctdistance=0.85,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 10}
    )

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    centre_circle = plt.Circle((0,0),0.42,fc='white')
    ax.add_artist(centre_circle)

    ax.set_title('Category Distribution', fontsize=14, pad=20)

    ax.axis('equal')  
    
    return fig


def plot_piecharts(sorted_cosine_scores_models):
    scores_list = []
    categories = st.session_state.categories.split(" ")
    index = 0
    for model in sorted_cosine_scores_models:
        scores_list.append(sorted_cosine_scores_models[model])
        index += 1

    if len(sorted_cosine_scores_models) == 2:
        fig, (ax1, ax2) = plt.subplots(2)

        categories_sorted = [
            categories[scores_list[0][index][0]] for index in range(len(scores_list[0]))
        ]
        sorted_scores = np.array(
            [scores_list[0][index][1] for index in range(len(scores_list[0]))]
        )
        ax1.pie(sorted_scores, labels=categories_sorted, autopct="%1.1f%%")

        categories_sorted = [
            categories[scores_list[1][index][0]] for index in range(len(scores_list[1]))
        ]
        sorted_scores = np.array(
            [scores_list[1][index][1] for index in range(len(scores_list[1]))]
        )
        ax2.pie(sorted_scores, labels=categories_sorted, autopct="%1.1f%%")

    st.pyplot(fig)


def plot_alatirchart(sorted_cosine_scores_models):
    models = list(sorted_cosine_scores_models.keys())
    tabs = st.tabs(models)
    figs = {}
    for model in models:
        figs[model] = plot_piechart_helper(sorted_cosine_scores_models[model])

    for index in range(len(tabs)):
        with tabs[index]:
            st.pyplot(figs[models[index]])

# Task I: Compute Cosine Similarity
def cosine_similarity(x, y):
    """
    Exponentiated cosine similarity
    1. Compute cosine similarity
    2. Exponentiate cosine similarity
    3. Return exponentiated cosine similarity
    (20 pts)
    """
    dot_product = np.dot(x, y)
    norm_x = la.norm(x)
    norm_y = la.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0.0  # Handle zero vectors to avoid division by zero
    cos_sim = dot_product / (norm_x * norm_y)
    return np.exp(cos_sim)

# Task II: Average Glove Embedding Calculation
def averaged_glove_embeddings_gdrive(sentence, word_index_dict, embeddings, model_type):
    """
    Get averaged glove embeddings for a sentence
    1. Split sentence into words
    2. Get embeddings for each word
    3. Sum embeddings for each word
    4. Divide by number of words
    5. Return averaged embeddings
    (30 pts)
    """
    model_dim = int(model_type.split('d')[0])
    words = sentence.split()
    avg_embedding = np.zeros(model_dim)
    if not words:
        return avg_embedding
    for word in words:
        word_embed = get_glove_embeddings(word, word_index_dict, embeddings, model_type)
        avg_embedding += word_embed
    avg_embedding /= len(words)
    return avg_embedding

# Task III: Sort the cosine similarity
def get_sorted_cosine_similarity(embeddings_metadata):
    """
    Get sorted cosine similarity between input sentence and categories
    Steps:
    1. Get embeddings for input sentence
    2. Get embeddings for categories (update if not found)
    3. Compute cosine similarity between input and categories
    4. Sort cosine similarities
    5. Return sorted cosine similarities
    (50 pts)
    """
    categories = st.session_state.categories.split(" ")
    cosine_sim = {}
    if embeddings_metadata["embedding_model"] == "glove":
        word_index_dict = embeddings_metadata["word_index_dict"]
        embeddings = embeddings_metadata["embeddings"]
        model_type = embeddings_metadata["model_type"]

        input_embedding = averaged_glove_embeddings_gdrive(
            st.session_state.text_search,
            word_index_dict,
            embeddings, model_type
        )
        
        # Compute cosine similarity for each category
        for idx, category in enumerate(categories):
            cat_embed = get_glove_embeddings(category, word_index_dict, embeddings, model_type)
            sim = cosine_similarity(input_embedding, cat_embed)
            cosine_sim[idx] = sim
        
    else:
        model_name = embeddings_metadata.get("model_name", "")
        if f"cat_embed_{model_name}" not in st.session_state:
            get_category_embeddings(embeddings_metadata)
        
        category_embeddings = st.session_state[f"cat_embed_{model_name}"]
        
        input_embedding = get_sentence_transformer_embeddings(
            st.session_state.text_search, model_name=model_name
        )
        
        for idx, category in enumerate(categories):
            if category not in category_embeddings:
                # Update missing category embedding
                category_embeddings[category] = get_sentence_transformer_embeddings(category, model_name=model_name)
            cat_embed = category_embeddings[category]
            sim = cosine_similarity(input_embedding, cat_embed)
            cosine_sim[idx] = sim
    
    # Sort scores in descending order
    sorted_scores = sorted(cosine_sim.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


if __name__ == "__main__":

    st.sidebar.title("Model Configuration")
    st.sidebar.markdown(
    """
    GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Pretrained on 
    2 billion tweets with vocabulary size of 1.2 million. Download from [Stanford NLP](http://nlp.stanford.edu/data/glove.twitter.27B.zip). 

    Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. *GloVe: Global Vectors for Word Representation*.
    """
    )

    st_model = st.sidebar.selectbox(
        "Sentence Transformer Model",
        options=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-mpnet-base-dot-v1",
            "paraphrase-multilingual-mpnet-base-v2"
        ],
        index=0,
        help="Select pretrained sentence transformer model"
    )

    model_type = st.sidebar.selectbox(
        "GloVe Dimension", 
        ("25d", "50d", "100d"), 
        index=1,
        help="Select dimension for GloVe embeddings"
    )

    st.title("Semantic Search Demo")

    if "categories" not in st.session_state:
        st.session_state.categories = "Flowers Colors Cars Weather Food"
    if "text_search" not in st.session_state:
        st.session_state.text_search = "Roses are red, trucks are blue, and Seattle is grey right now"

    st.subheader("Categories (space-separated)")
    st.text_input(
        label="Categories",
        key="categories",
        value=st.session_state.categories
    )
    
    st.subheader("Input Sentence")
    st.text_input(
        label="Your input",
        key="text_search",
        value=st.session_state.text_search
    )

    embeddings_path = f"embeddings_{model_type}_temp.npy"
    word_index_dict_path = f"word_index_dict_{model_type}_temp.pkl"
    if not os.path.isfile(embeddings_path) or not os.path.isfile(word_index_dict_path):
        with st.spinner(f"Downloading GloVe-{model_type} embeddings..."):
            download_glove_embeddings_gdrive(model_type)

    word_index_dict, embeddings = load_glove_embeddings_gdrive(model_type)

    if st.session_state.text_search.strip():

        glove_metadata = {
            "embedding_model": "glove",
            "word_index_dict": word_index_dict,
            "embeddings": embeddings,
            "model_type": model_type,
        }

        transformer_metadata = {
            "embedding_model": "transformers", 
            "model_name": st_model
        }

        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner(f"Processing GloVe-{model_type}..."):
                sorted_glove = get_sorted_cosine_similarity(glove_metadata)
        
        with col2:
            with st.spinner(f"Processing {st_model}..."):
                sorted_transformer = get_sorted_cosine_similarity(transformer_metadata)

        st.subheader(f"Results for: '{st.session_state.text_search}'")
        plot_alatirchart({
            f"Sentence Transformer ({st_model})": sorted_transformer,
            f"GloVe-{model_type}": sorted_glove
        })

        st.markdown("---")
        st.caption("Developed by [Xinghao Chen](https://www.linkedin.com/in/cxh42/) | "
                   "Model credits: [Sentence Transformers](https://www.sbert.net/) |"
                   "[GloVe](https://nlp.stanford.edu/projects/glove/)")
```