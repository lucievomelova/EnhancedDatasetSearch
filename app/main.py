from flask import Flask, render_template, request
from pipeline import SearchPipeline
import streamlit as st
import yaml


app = Flask(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

search_pipeline = SearchPipeline(config)


@app.route('/', methods=['GET', 'POST'])
def home():
    query = ''
    table_html = None
    streamlit_chatbot()

    if request.method == 'POST':
        query = request.form.get('query', '').strip().lower()
        if query:
            results = search_pipeline.run(query)
            if results is not None:
                table_html = results.to_html(index=False, escape=False)
            else:
                table_html = "<p>No matching results found.</p>"

    return render_template("home.html", query=query, table=table_html)


def streamlit_chatbot():

    # Initialize session state
    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False

    st.title("My Streamlit App")

    # Normal app content
    st.write("Here's the main part of your app.")
    if st.button("💬 Open Chatbot"):
        st.session_state.show_chat = True

    # Simulated popup
    if st.session_state.show_chat:
        with st.container():
            st.markdown(
                """
                <div style='position: fixed; bottom: 20px; right: 20px;
                            width: 300px; height: 400px;
                            background-color: white; border: 2px solid #ccc;
                            border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                            padding: 10px; z-index: 100;'>
                """,
                unsafe_allow_html=True
            )
            st.write("🤖 **Chatbot**")
            user_input = st.text_input("You:", key="chat_input")
            if user_input:
                st.write(f"**Bot:** You said '{user_input}'")
            if st.button("❌ Close"):
                st.session_state.show_chat = False
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    app.run(debug=True)