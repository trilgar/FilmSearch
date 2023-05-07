import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Press the green button in the gutter to run the script.

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", force=True)


# model_path = "fasttext/cc.en.300.bin"
# model = fasttext.load_model(model_path)

# engine = EmbeddingSearch.from_texts(df["keywords"], model.get_sentence_vector)
@st.cache_data
def startup():
    PATH = Path("data")

    print('Started indexing')

    df_keywords = pd.read_csv(PATH / "keywords.csv")
    df_keywords["keywords"] = df_keywords["keywords"].apply(yaml.safe_load)

    df_meta = pd.read_csv(PATH / "movies_metadata.csv")
    df_meta = df_meta[df_meta["id"].str.isnumeric()]
    df_meta["id"] = df_meta["id"].astype("int")

    df = df_meta.merge(df_keywords, on="id", how="left")
    df = df.dropna(subset="keywords").copy()
    df["keywords"] = df["keywords"].apply(lambda x: " ".join(x["name"] for x in x))

    print('Completed indexing')
    return df


def main():
    # Define the search box widget
    query = st.text_input("Enter a search term:")
    df = startup()

    if st.button("Search"):
        # Get the search results from your search engine
        # results = engine.get_closest(query, MAX_RESULTS)
        f = open('test.json')

        results = json.load(f)

        st.write(df.iloc[[x["id"] for x in results]])


if __name__ == '__main__':
    main()
