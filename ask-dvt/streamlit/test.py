import uuid
import streamlit as st
from datetime import datetime
from idx_db_component.frontend import idxdb_component


def main():
    params = {
        'operation': 'update',
        'sessionid': str(uuid.uuid4()),
        'datetime': str(datetime.now().strftime("%H:%M:%S, %d %b %Y"))
    }

    print(datetime.now())
    result = idxdb_component(key="test", params=params)
    print(datetime.now())
    print("===========================================\n", result, "\n===========================================")

    if result:
        if "loading" in result and result["loading"]:
            st.info("Loading data from IndexedDB...")
        elif "records" in result:
            st.success("Data loaded successfully!")
            st.write(result["records"])
        elif "error" in result:
            st.error(f"Error: {result['error']}")


# if __name__ == '__main__':
#     main()