import asyncio
import os
import threading
import uuid

import yaml
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from app.chatbot import Chatbot
from app.pipeline import SearchPipeline
from app.ui_utils import get_filters_for_results, get_similar_datasets_with_preview_text
from data_processing.data_catalog import DataCatalog
from data_processing.database import Database
from data_processing.knowledge_graph import KnowledgeGraph
from data_processing.NKOD.knowledge_graph import NkodKnowledgeGraph
from data_processing.NKOD.nkod import NkodDataCatalog
from utils import setup_logger

logger = setup_logger(__name__)

# create event loop
loop = asyncio.new_event_loop()

def start_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_loop, daemon=True).start()

app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# setup LLM and embedding model
llm = Ollama(model=config['chatbot']['llm']['model_name'],
             context_window=config['chatbot']['llm']['context_length'],
             request_timeout=300)

Settings.llm = llm
Settings.embed_model = OllamaEmbedding(
    model_name=config['embedding']['model_name'],
    base_url=config['embedding']['base_url'],
    embed_batch_size=config['embedding']['embed_batch_size'],
)

search_pipeline_instances = {}  # dictionary to store search pipeline instances per session
chatbot_instances = {}  # dictionary to store chatbot instances per session
database: Database = Database(config)
data_catalog: DataCatalog = NkodDataCatalog(config, False)
knowledge_graph: KnowledgeGraph = NkodKnowledgeGraph(config["data_processing"]["knowledge_graph"], database)


def get_current_session_id():
    """Get current session id or create a new one if it doesn't exist for current session."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']  # return the id of the current session


def get_search_pipeline():
    """Get or create a search pipeline instance for the current session."""
    session_id = get_current_session_id()

    # create a new search pipeline instance for this session if it doesn't exist
    if session_id not in search_pipeline_instances:
        search_pipeline_instances[session_id] = SearchPipeline(config, llm, data_catalog, database)
    return search_pipeline_instances[session_id]


def get_chatbot():
    """Get or create a chatbot instance for the current session."""
    session_id = get_current_session_id()

    # create a new chatbot instance for this session if it doesn't exist
    if session_id not in chatbot_instances:
        pipeline = get_search_pipeline()
        chatbot_instances[session_id] = Chatbot(config, pipeline, data_catalog, llm)
    return chatbot_instances[session_id]


@app.route('/', methods=['GET', 'POST'])
def home():
    """Display the home page for the search engine."""
    return render_template("home.html", filters=data_catalog.get_filters_with_counts())


@app.route('/search', methods=['GET', 'POST'])
def search():
    """Display the page with search results."""
    pipeline = get_search_pipeline()
    applied_filters = {}

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        for key in request.form:
            if key.startswith("filter_"):
                category = key[len("filter_"):]  # get metadata category
                applied_filters[category] = request.form.getlist(key)
    else:
        query = request.args.get('query', '').strip()
        for key in request.args:
            if key.startswith("filter_"):
                category = key[len("filter_"):]  # get metadata category
                applied_filters[category] = request.form.getlist(key)

    results = None
    filters = {}
    if query:
        logger.info(f"Session: {session}")
        future = asyncio.run_coroutine_threadsafe(
            pipeline.run(query, applied_filters),
            loop
        )
        results = future.result()
        filters = get_filters_for_results(results)

    for filter_cat, cat_data in filters.items():
        if filter_cat not in applied_filters:
            applied_filters[filter_cat] = []  # set applied filters to empty for skipped filter categories
        # keep only applied filters that are present in the results
        applied_filters[filter_cat] = [a for a in applied_filters[filter_cat] if a in cat_data["value_counts"].keys()]

    return render_template("search_results.html", query=query, results=results, filters=filters, applied_filters=applied_filters)


@app.route('/dataset_detail')
def dataset_detail():
    """Display detailed view of a specific dataset by looking it up in extended_df."""
    dataset_url = request.args.get('source', '')
    dataset_info = data_catalog.get_dataset_by_url(dataset_url)

    if not dataset_info:
        return redirect(url_for('home'))
    similar_datasets = get_similar_datasets_with_preview_text(dataset_url,
                                                              dataset_info,
                                                              data_catalog,
                                                              knowledge_graph)
    distributions = dataset_info["distributions"]
    for i, d in enumerate(distributions, start=1):
        d["title"] = f"Distribution {i}" if d["title"] is None else d["title"]
    return render_template("dataset_detail.html",
                           dataset=dataset_info,
                           similar_datasets=similar_datasets,
                           distributions=distributions)


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
            chatbot_instances[session_id] = Chatbot(config, get_search_pipeline(), data_catalog)

    return jsonify({'success': True, 'message': 'Chatbot memory cleared.'})


@app.route('/chatbot/query', methods=['POST'])
def chatbot_query():
    """Handle chatbot queries using the Chatbot class."""
    data = request.get_json()
    user_query = data.get('query', '').strip()

    if not user_query:
        return jsonify({'success': False, 'response': 'Please type in a message.'})

    try:
        future = asyncio.run_coroutine_threadsafe(
            get_chatbot().react_to_message(user_query),
            loop
        )
        result = future.result()
        logger.info(f"Chatbot response: {result}")

        return jsonify({
            'success': True,
            'response': result
        })

    except Exception as e:
        import logging
        logger.error(f"Error in chatbot query: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'response': 'An error occurred while processing your query.'})


if __name__ == '__main__':
    app.run(debug=True)
