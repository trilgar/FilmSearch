import pandas as pd
import streamlit as st

# Press the green button in the gutter to run the script.
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

# Get the search results from your search engine
data = pd.DataFrame({
    'country': ['Japan', 'China', 'Thailand', 'France', 'Belgium', 'South Korea'],
    'capital': ['Tokyo', 'Beijing', 'Bangkok', 'Paris', 'Brussels', 'Seoul']
})

gd = GridOptionsBuilder.from_dataframe(data)
gd.configure_column("Relevance",
                    header_name='Relevance',
                    editable=True,
                    cellEditor='agSelectCellEditor',
                    cellEditorParams={
                        'values': ["", "Relevant", "NotRelevant"]
                    })

grid_options = gd.build()
# Display the DataFrame with radio buttons using AgGrid
grid_response = AgGrid(
    data=data,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
    allow_unsafe_jscode=True
)

if st.button("Recalculate with relevance"):
    updated_data = grid_response['data']
    print(updated_data.to_dict('records'))
