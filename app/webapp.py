from flask import Flask, render_template, request
from pipeline import SearchPipeline
import yaml


app = Flask(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

search_pipeline = SearchPipeline(config)


@app.route('/', methods=['GET', 'POST'])
def home():
    query = ''
    results = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip().lower()
        if query:
            results = search_pipeline.run(query)

    return render_template("home.html", query=query, results=results)


if __name__ == '__main__':
    app.run(debug=True)