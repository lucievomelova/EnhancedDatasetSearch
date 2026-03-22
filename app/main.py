import asyncio
import nest_asyncio

from flask import Flask, render_template, request, redirect, url_for
from app.pipeline import SearchPipeline
import yaml
from urllib.parse import unquote

from data_processing.knowledge_graph import get_similar_datasets
from app.ui_utils import get_common_metadata

nest_asyncio.apply()  # allow nested event loops - each pipeline run would create a new event loop otherwise
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)  # single event loop for the whole app

app = Flask(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# lazy initialization to avoid event loop issues
search_pipeline = None

def get_search_pipeline():
    global search_pipeline
    if search_pipeline is None:
        search_pipeline = SearchPipeline(config)
    return search_pipeline


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


@app.route('/dataset/<path:dataset_url>')
def dataset_detail(dataset_url):
    """Display detailed view of a specific dataset by looking it up in extended_df."""
    dataset_url = unquote(dataset_url)
    pipeline = get_search_pipeline()
    dataset_info = pipeline.dataset_portal.get_dataset_by_url(dataset_url)

    if not dataset_info:
        return redirect(url_for('home'))

    similar_datasets_raw = get_similar_datasets(dataset_url, top_k=5)

    similar_datasets = {}
    for sim_category, url_score_list in similar_datasets_raw.items():
        similar_datasets[sim_category] = []
        for url, _ in url_score_list:
            sim_dataset = pipeline.dataset_portal.get_dataset_by_url(url)
            if sim_dataset:
                text_preview = sim_dataset['text'][:sim_dataset['text'].find(".")+1]
                if len(text_preview) > 200:  # we dont want too long description text preview
                    index = text_preview[:200].rfind(" ")
                    text_preview = text_preview[:index] + "..."
                similar_dataset_info = {
                    'title': sim_dataset['title'],
                    'url': url,
                    'text_preview': text_preview,
                }
                if sim_category == "overall":
                    metadata_categories = ["keywords", "themes", "categories", "region", "time_period"]
                    common_metadata = get_common_metadata(metadata_categories, dataset_info["metadata"], sim_dataset["metadata"])
                    similar_dataset_info["common_metadata"] = common_metadata
                elif sim_category != "description":
                    common_metadata = get_common_metadata([sim_category], dataset_info["metadata"], sim_dataset["metadata"])
                    similar_dataset_info["common_metadata"] = common_metadata
                similar_datasets[sim_category].append(similar_dataset_info)

    return render_template("dataset_detail.html", dataset=dataset_info, similar_datasets=similar_datasets, query='')


if __name__ == '__main__':
    app.run(debug=True)
