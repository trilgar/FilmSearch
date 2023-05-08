import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
import fasttext

# Press the green button in the gutter to run the script.
from database import FilmsDB
from models import EmbeddingSearch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

returned_columns = ['_id', 'title', 'keywords']


@st.cache_resource
def startup():
    logging.debug('Started model preparation')
    database = FilmsDB()
    model_path = "fasttextmodel/cc.en.300.bin"
    model = fasttext.load_model(model_path)

    engine = EmbeddingSearch.from_database(database, model.get_sentence_vector)
    logging.debug('Model preparation completed')
    return database, engine


def main():
    database, search_engine = startup()

    # Define the search box widget
    query = st.text_input("Enter search request:")
    num_results = st.number_input("Number of results to return:", value=10)

    if st.button("Search"):
        # Get the search results from your search engine
        results = search_engine.get_closest(query, num_results)
        data = pd.DataFrame(list(database.get_by_ids([x["id"] for x in results.result])))
        st.write(data[returned_columns])


if __name__ == '__main__':
    main()
