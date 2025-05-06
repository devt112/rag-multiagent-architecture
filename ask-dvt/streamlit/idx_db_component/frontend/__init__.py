import os
import streamlit.components.v1 as components

script_directory = os.path.dirname(os.path.abspath(__file__))
# target_directory = os.path.join(script_directory, 'idx_db_component', 'frontend')

idxdb_component = components.declare_component(
    name='idxdb_component',
    path=script_directory
)