import asyncio
import os

import nest_asyncio
import pandas as pd
import uuid

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from app.pipeline import SearchPipeline
from app.chatbot import Chatbot
import yaml
from urllib.parse import unquote

from data_processing.knowledge_graph import get_similar_datasets
from app.ui_utils import get_common_metadata
from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

nest_asyncio.apply()  # allow nested event loops - each pipeline run would create a new event loop otherwise
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)  # single event loop for the whole app

app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

llm = Ollama(model=config['chatbot']['llm']['model_name'], context_window=config['chatbot']['llm']['context_length'],
                  request_timeout=300)
Settings.llm = llm
Settings.embed_model = OllamaEmbedding(
    model_name=config['embedding']['model_name'],
    base_url=config['embedding']['base_url'],
    embed_batch_size=config['embedding']['embed_batch_size'],
)

# lazy initialization to avoid event loop issues
search_pipeline = None
chatbot_instances = {}  # dictionary to store chatbot instances per session

def get_search_pipeline():
    global search_pipeline
    if search_pipeline is None:
        search_pipeline = SearchPipeline(config, llm)
    return search_pipeline

def get_chatbot():
    """Get or create a chatbot instance for the current session."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    
    # create a new chatbot instance for this session if it doesn't exist
    if session_id not in chatbot_instances:
        pipeline = get_search_pipeline()
        chatbot_instances[session_id] = Chatbot(config, pipeline, llm)
    
    return chatbot_instances[session_id]

search_pipeline = get_search_pipeline()


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            return redirect(url_for('search', query=query))
    return render_template("home.html")


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
    else:
        query = request.args.get('query', '').strip()

    results = None
    if query:
        pipeline = get_search_pipeline()
        results = loop.run_until_complete(pipeline.run(query))

    return render_template("search_results.html", query=query, results=results)


@app.route('/dataset_detail')
def dataset_detail():
    """Display detailed view of a specific dataset by looking it up in extended_df."""
    dataset_url = request.args.get('source', '')
    pipeline = get_search_pipeline()
    dataset_info = pipeline.data_catalog.get_dataset_by_url(dataset_url)

    if not dataset_info:
        return redirect(url_for('home'))

    similar_datasets_raw = get_similar_datasets(dataset_url, config["data_processing"]["knowledge_graph"]["top_k"])

    similar_datasets = {}
    for sim_category, url_score_list in similar_datasets_raw.items():
        similar_datasets[sim_category] = []
        for url, _ in url_score_list:
            sim_dataset = pipeline.data_catalog.get_dataset_by_url(url)
            if sim_dataset:
                text_preview = ""
                if pd.notna(sim_dataset['text']):
                    print(sim_dataset['text'])
                    text_preview = sim_dataset['text']
                    if len(text_preview) > 200:  # too long description text preview -> take just first sentence
                        text_preview = sim_dataset['text'][:sim_dataset['text'].find(".")+1]

                    # first sentence still too long or there is no "." char in the description
                    if len(text_preview) > 200 or len(text_preview) == 0:
                        index = sim_dataset['text'][:200].rfind(" ")
                        text_preview = sim_dataset['text'][:index]
                        if len(text_preview) > 0:
                            text_preview += "..."  # if the preview is not empty, show it was cut off by appending ...
                similar_dataset_info = {
                    'title': sim_dataset['title'],
                    'url': url,
                    'text_preview': text_preview,
                }
                if sim_category == "overall":
                    metadata_categories = ["keywords", "themes", "categories", "region", "time_periods"]
                    common_metadata = get_common_metadata(metadata_categories, dataset_info["metadata"], sim_dataset["metadata"])
                    similar_dataset_info["common_metadata"] = common_metadata
                elif sim_category != "description":
                    common_metadata = get_common_metadata([sim_category], dataset_info["metadata"], sim_dataset["metadata"])
                    similar_dataset_info["common_metadata"] = common_metadata
                similar_datasets[sim_category].append(similar_dataset_info)

    return render_template("dataset_detail.html", dataset=dataset_info, similar_datasets=similar_datasets, query='')


@app.route('/chatbot')
def chatbot():
    """Display the chatbot page."""
    return render_template("chatbot.html")


@app.route('/chatbot/reset', methods=['POST'])
def reset_chatbot():
    """Reset the chatbot memory for the current session."""
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in chatbot_instances:
            del chatbot_instances[session_id]
            chatbot_instances[session_id] = Chatbot(config, get_search_pipeline())

    return jsonify({'success': True, 'message': 'Chatbot memory cleared.'})


@app.route('/chatbot/query', methods=['POST'])
def chatbot_query():
    """Handle chatbot queries using the Chatbot class."""
    data = request.get_json()
    user_query = data.get('query', '').strip()
    
    if not user_query:
        return jsonify({'success': False, 'response': 'Please type in a message.'})
    
    try:
        result = loop.run_until_complete(get_chatbot().react_to_message(user_query))
        print(result)
        
        return jsonify({
            'success': True,
            'response': result
        })
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in chatbot query: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'response': 'An error occurred while processing your query.'})


if __name__ == '__main__':
    app.run(debug=True)
