import logging

import fasttext
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

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


if "search_results" not in st.session_state:
    st.session_state.search_results = 'none'
    logging.debug("Initialized session parameters: search_results")
if "aggrid" not in st.session_state:
    st.session_state.aggrid = 'none'
    logging.debug("Initialized session parameters: aggrid")


def main():
    database, search_engine = startup()

    # Define the search box widget
    query = st.text_input("Enter search request:", key="searchbar")
    num_results = st.number_input("Number of results to return:", value=10, key='result_size')

    if st.button("Search") or st.session_state.search_results != 'none':
        if st.session_state.search_results == 'none':
            # Get the search results from your search engine
            search_results = search_engine.get_closest(query, num_results)
            st.session_state.search_results = search_results
        else:
            search_results = st.session_state.search_results

        data = pd.DataFrame(list(database.get_by_ids([x["id"] for x in search_results.result])))
        data['_id'] = data['_id'].astype(str)
        data = data[returned_columns]
        gd = GridOptionsBuilder.from_dataframe(data)
        gd.configure_column("relevance",
                            header_name='relevance',
                            editable=True,
                            cellEditor='agSelectCellEditor',
                            cellEditorParams={
                                'values': ["", "Relevant", "Not Relevant"]
                            })
        grid_options = gd.build()
        grid_response = AgGrid(
            data=data,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            width=700,
            allow_unsafe_jscode=True,
            key='result'
        )
        st.session_state.aggrid = grid_response

        def update_weights():
            logging.debug("retrieving marked data")
            updated_data = grid_response['data']
            relevant_data = updated_data[['_id', 'relevance']]
            # fill nan values with None
            relevant_data['relevance'] = relevant_data['relevance'].fillna(value='None')
            relevant_data["relevance"] = relevant_data["relevance"].replace(
                {'Not Relevant': False, 'Relevant': True, 'None': None})
            logging.debug(relevant_data.to_dict('records'))
            # todo rerank functionality

        st.button("Recalculate with relevance", on_click=update_weights)


if __name__ == '__main__':
    main()
