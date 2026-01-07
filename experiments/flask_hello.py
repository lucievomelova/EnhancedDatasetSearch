from flask import Flask, render_template, request
from rdflib import Graph, Literal
from rdflib.plugins.stores.sparqlstore import SPARQLStore
import ollama

endpoint = "https://data.gov.cz/sparql"
g = Graph(SPARQLStore(endpoint))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ''
    results = []

    if request.method == 'POST':
        user_input = request.form.get('query_term', '').strip()
        limit = request.form.get('limit', '10')
        endpoint = "https://data.gov.cz/sparql"
        g = Graph(SPARQLStore(endpoint))
        user_literal = Literal(user_input)  # convert user input to an RDF literal (prevents injection)

        query = f"""
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX ds: <https://data.gov.cz/zdroj/datové-sady/00025593/>
        PREFIX lsgov: <https://slovník.gov.cz/legislativní/sbírka/111/2009/pojem/>

        SELECT ?poskytovatel ?nazev
        WHERE {{
          ?poskytovatel lsgov:má-název-orgánu-veřejné-moci ?nazev .
          FILTER(CONTAINS(LCASE(?nazev), LCASE({user_literal.n3()})))
        }}
        LIMIT {limit}
        """

        try:
            for row in g.query(query):
                poskytovatel, nazev = row
                results.append({
                    "poskytovatel": str(poskytovatel),
                    "nazev": str(nazev)
                })
        except Exception as e:
            results = [{"error": str(e)}]

    return render_template("index.html", results=results, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)


query = """
You are a helpful AI assistant on a data catalog web page. Users are searching for datasets for their research, 
work, projects, etc. They input a search query and your job is to expand the query using synonyms, related terms, 
and relevant concepts, so that the search results are more comprehensive and useful.
Be absolutely sure to return just the expanded query in natural language without any additional commentary.
Expand the following user query: 
"""


@app.route('/ollama', methods=['GET', 'POST'])
def ollama_page():
    message = ""
    if request.method == 'POST':
        user_input = request.form.get('text_input', '').strip()

        result = ollama.generate(model='tinyllama:1.1b', prompt=f'{query} + {user_input}')
        return render_template('ollama.html', result=result["response"])
    return render_template('ollama.html', result="")
