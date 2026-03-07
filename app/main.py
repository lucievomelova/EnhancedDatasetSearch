import asyncio
import nest_asyncio

from flask import Flask, render_template, request, redirect, url_for
from app.pipeline import SearchPipeline
import yaml


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
    results = None
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
    else:
        query = request.args.get('query', '').strip()

    if query:
        pipeline = get_search_pipeline()
        results = loop.run_until_complete(pipeline.run(query))
    return render_template("search_results.html", query=query, results=results)


if __name__ == '__main__':
    app.run(debug=True)
