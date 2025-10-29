from flask import Flask, render_template, request
from pipeline import run


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    query = ''
    table_html = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip().lower()
        if query:
            results = run(query)
            if results is not None:
                table_html = results.to_html(index=False, escape=False)
            else:
                table_html = "<p>No matching results found.</p>"

    return render_template("home.html", query=query, table=table_html)

if __name__ == '__main__':
    app.run(debug=True)