import os, sys, time, json, datetime, random, app_config, uuid, logging
import pandas as pd
import streamlit as st
from io import StringIO
import concurrent.futures
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from memory import MemoryManager
from state import ToolBaseSchema
from langgraph.types import Command
from commons.utils import Utilities
from configuration import Configuration
from commons.processing import Processor
from commons.logger import CustomLogger
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, AIMessage

# Configuration
DEFAULT_CONFIG = {
    "configurable": {
        "thread_id": "46252",
        "num_similar_docs": "8",
        "similarity_threshold": "0.7",
        "query_weight": "0.7"
    }
}

class StreamlitApp:
    def __init__(self):
        st.set_page_config(layout="wide", initial_sidebar_state="auto", menu_items=None)

        my_logger = CustomLogger(
            level=logging.DEBUG,
            log_format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
        self.logger = my_logger.get_logger()
        
        if st.session_state.get("memory_manager", None) is None:
            st.session_state.memory_manager = MemoryManager()
        
        if st.session_state.get("session_id", None) is None and st.session_state.get("last_thread_id", None) is None:
            st.session_state.memory_manager.initialize_session_state()
            
        # for header in st.context.headers:
        #     print(header, st.context.headers[header])

    def run(self):
        if not st.session_state.get("chat_input_disabled", False):
            if st.session_state.get("session_id", None) is not None:
                self._load_chat_history(st.session_state.session_id)
                
        self._setup_interface_()
                
        if st.session_state.get("stage") == "user":
            self._handle_user_input()
        elif st.session_state.get("stage") == "validate":
            self._handle_validation_stage()
                
    def _setup_interface_(self):
        self._display_sidebar()
        
        with st.container(key="dvt-info"):             
            image_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "images", "ask-dvt-icon.jpg")
            st.image(image_file_path, width=250, output_format="PNG")
            
        # st.markdown('<div style="height:10vh;"></div>', unsafe_allow_html=True)

    def _display_sidebar(self):
        def _load_history_for_thread(session_id, thread_id):
            st.session_state.history = []
            st.session_state.chat_input_disabled = True
            self._load_chat_history(session_id=session_id, thread_id=thread_id, new_session=False)
        
        @st.dialog("dvt AI Chatbot: Your Intelligent Assistant", width="large")
        def show_app_description():
            st.write(app_config.APP_DESCRIPTION)
            
        @st.dialog("dvt AI Chatbot: Your Intelligent Assistant", width="large")
        def show_user_guide():
            st.write(app_config.APP_USER_GUIDE)

        with st.sidebar:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container():
                if st.button("APP CAPABILITIES", type="secondary", use_container_width=True):
                    show_app_description()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.container():   
                if st.button("USER GUIDE", type="secondary", use_container_width=True):
                    show_user_guide()
                    
            st.markdown(f"<br>", unsafe_allow_html=True)
            
            if st.button("NEW SESSION", type="primary", use_container_width=True):
                st.session_state.memory_manager.create_new_session()
                st.session_state.chat_input_disabled = False
                
            st.markdown("<br>", unsafe_allow_html=True)

            with st.container(key="kb-settings"):
                col1, col2 = st.columns(2)
                with col1:
                    similarity_threshold = st.number_input(
                        "Similarity Threshold",
                        value=float(0.7),
                        max_value=float(0.9),
                        min_value=float(0.2),
                        step=float(0.1),
                        help="Threshold for document similarity.",
                    )
                with col2:
                    similar_results_count = st.number_input(
                        "Similar Results Count",
                        value=8,
                        max_value=10,
                        min_value=5,
                        step=1,
                        help="Number of similar documents to retrieve.",
                    )
                with col1:
                    query_weight = st.number_input(
                        "Query Weight",
                        value=float(0.7),
                        min_value=float(0.1),
                        max_value=float(0.9),
                        step=float(0.1),
                        help="Weight given to the user query.",
                    )
                with col2:
                    st.number_input(
                        "Response Weight",
                        value=round(float(1 - query_weight), 1),
                        disabled=True,
                        help="Weight given to the response (calculated).",
                    )
            
            st.subheader("HISTORY", divider="gray")
            st.markdown("<br>", unsafe_allow_html=True)
            
            thread_history = st.session_state.memory_manager.get_thread_history()
            if thread_history:
                _, st.session_state.last_thread_id, _, _ = thread_history[0]
                with st.container(key="th-history"):
                    for session_id, thread_id, latest_que, created_date in thread_history[1:]:
                        if created_date and latest_que:
                            thread = created_date + "-" + latest_que
                            st.button(thread, key=thread_id, type="tertiary", use_container_width=True, on_click=_load_history_for_thread, args=(session_id, thread_id,), help="Load this session history.")

        if not st.session_state.get("FormSubmitter:hil_form-Submit", False):
            st.session_state.user_config["configurable"]["data_source"] = None
            st.session_state.user_config["configurable"]["num_similar_docs"] = str(similar_results_count)
            st.session_state.user_config["configurable"]["similarity_threshold"] = str(similarity_threshold)
            st.session_state.user_config["configurable"]["query_weight"] = str(query_weight)

    def _load_chat_history(self, session_id=None, thread_id=None, **kwargs):
        def reset_session():
            st.session_state.history = []
            st.session_state.chat_input_disabled = False

        session_id = st.session_state.session_id if session_id is None else session_id
            
        with st.spinner("Loading chat history..."):
            history_future = st.session_state.memory_manager.load_history(session_id, thread_id)  # Get the Future
            history_from_db = history_future.result()  # Block and get the result

        if history_from_db:
            st.session_state.history = [{"role": role, "content": {"pending": content}} for role, content in history_from_db]
            if len(st.session_state.history) > 0:
                # st.session_state.last_history_sessionid = session_id
                st.markdown('<div style="height:10vh;"></div>', unsafe_allow_html=True)
                self._display_chat_history()
                self.formatted_chat_history = self._format_chat_history_for_agent()
                if st.session_state.get("agent_state", None) and hasattr(st.session_state.get("agent_state"), "next"):
                    self._human_in_the_loop()
                
        if "new_session" in kwargs and not kwargs["new_session"]:
            col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
            with col_t2:
                st.button("Return to Active Session", type="primary", on_click=reset_session, help="Click to Return to active session.")

    def _display_chat_history(self):
        """Displays the chat history from the session state."""
        if "history" in st.session_state:
            for message in st.session_state.history:
                if message["role"] is not None and message["role"]:
                    msg_content = message.get("content", {})
                    json_msg_content = Utilities.extract_json_from_string(msg_content.get("pending", ""))
                    with st.chat_message(message["role"]):
                        if isinstance(json_msg_content, dict):
                            self.display_airesponse_details(Processor._to_obj(json_msg_content))
                            st.markdown("<hr id='chat_divider'>", unsafe_allow_html=True)
                        else: st.write(msg_content.get("pending"))
            
    def _format_chat_history_for_agent(self):
        formatted_messages = []
        if len(st.session_state.history) > 10:
            old_history = st.session_state.history[:-10]
            old_history_text = []
            for msg in old_history:
                role = msg.get("role")
                content = msg.get("content", {}).get("pending", "")
                if role == "user":
                    old_history_text.append(f"User: {content}")
                elif role == "assistant":
                    content = Utilities.extract_json_from_string(content)["response"]
                    old_history_text.append(f"Assistant: {content}")
            old_history_text = "\n".join(old_history_text)

            gemini_model: ChatVertexAI = Configuration.query_model
            summary_prompt = f"Summarize the following conversation history:\n{old_history_text}"
            summary_messages = [HumanMessage(content=summary_prompt)]
            summary = gemini_model.invoke(summary_messages).content
            summary = f"Summary of earlier conversation: {summary}"
            formatted_messages.append(f"Assistant: {summary}")
            recent_history = st.session_state.history[-10:]
        else:
            recent_history = st.session_state.history
            
        for message in recent_history:
            role = message.get("role")
            content = message.get("content", {}).get("pending", "")
            if role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                content = Utilities.extract_json_from_string(content)["response"]
                formatted_messages.append(f"Assistant: {content}")
                
        return formatted_messages

    def _handle_user_input(self):
        if user_query := st.chat_input("How can I help you today", disabled=st.session_state.chat_input_disabled):
            self.queid = str(uuid.uuid4())
            
            if st.session_state.get("agent_response", None): del st.session_state["agent_response"] 
            if st.session_state.get("agent_state", None): del st.session_state["agent_state"]
            
            if st.session_state.get("last_thread_id", None) is not None and st.session_state.get("session_id", None) is not None:
                self.logger.info("Session ID: %s | Thread ID: %s | User Query: %s", st.session_state.session_id, st.session_state.last_thread_id, user_query)
            
            if st.session_state.get("history", None) is None or len(st.session_state.history) == 0:
                st.markdown('<div style="height:10vh;"></div>', unsafe_allow_html=True)
            
            with st.chat_message("user"):
                st.write(user_query)

            with st.spinner("Processing..."):
                start_time = time.time()
                user_config = st.session_state.user_config
                user_config["configurable"]["thread_id"] = str(uuid.uuid4())
                st.session_state.user_config = user_config
                
                self.logger.info("Session ID: %s | Thread ID: %s | NEW QUERY | user_config: %s", st.session_state.session_id, st.session_state.last_thread_id, st.session_state.user_config)
                self.logger.info("Session ID: %s | Thread ID: %s | NEW QUERY | chat_history: %s", st.session_state.session_id, st.session_state.last_thread_id, self.formatted_chat_history)
                
                try:
                    st.session_state.agent_response, st.session_state.agent_state = st.session_state.memory_manager.processor.process_user_query(
                        user_query, st.session_state.user_config, self.formatted_chat_history
                    )
                except Exception as e:
                    self.logger.error(f"Error processing user query: {e}", exc_info=True)
                    st.session_state.agent_response = None
                    st.session_state.agent_state = None
                
                end_time = time.time()
                response_time = end_time - start_time

                if st.session_state.agent_response is not None:
                    if hasattr(st.session_state.agent_response, "info"):
                        st.session_state.agent_response.info["response_time"] = (
                            f"Response time: {response_time:.2f} seconds"
                        )
                        is_approved_response = st.session_state.agent_response.info.get("is_preapproved", False)

                    if any(hasattr(st.session_state.agent_response, attr) for attr in ["response", "code", "datatable"]):
                        store_future = st.session_state.memory_manager.store_messages([
                            {
                                "session_id": st.session_state.get("session_id"),
                                "thread_id": st.session_state.get("last_thread_id"),
                                "role": "user",
                                "queid": self.queid,
                                "content": user_query
                            },
                            {
                                "session_id": st.session_state.get("session_id"),
                                "thread_id": st.session_state.get("last_thread_id"),
                                "role": "assistant",
                                "queid": self.queid,
                                "content": json.dumps(st.session_state.agent_response.model_dump())
                            },
                        ])
                    
                    self._handle_validation_stage(is_approved_response, False)
                else:
                    with st.chat_message("assistant"):
                        st.write("""I'm having trouble getting a response right now. Before retrying, you could try:
                            * Simplifying your query.
                            * Checking for any typos.
                            * Trying again in a few moments.
                            I hope this helps!"""
                        )

    def _human_in_the_loop(self):        
        @st.dialog("Action Executed Successfully", width="small")
        def show_hil_action_result(message):
            st.markdown(message, unsafe_allow_html=True)
            
        def hil_action(hil_form_elements, formatted_chat_history):
            hil_payload = {}
            for element in hil_form_elements:
                hil_payload[element] = st.session_state.get(element, None)

            if len(hil_payload) > 0:
                self.logger.info("Session ID: %s | Thread ID: %s | HUMAN IN THE LOOP | user_config: %s", st.session_state.session_id, st.session_state.last_thread_id, st.session_state.user_config)
                self.logger.info("Session ID: %s | Thread ID: %s | HUMAN IN THE LOOP | chat_history: %s", st.session_state.session_id, st.session_state.last_thread_id, formatted_chat_history)
                st.session_state.agent_response, st.session_state.agent_state = st.session_state.memory_manager.processor.process_user_query(
                    Command(resume=hil_payload), 
                    st.session_state.user_config, 
                    formatted_chat_history
                )
                
                messages = Utilities.extract_tool_messages(st.session_state.agent_state.values["messages"])
                if messages.get("executor_result", {}).get("status", None) and messages.get("executor_result", {}).get("actions", None):
                    status = messages.get("executor_result", {}).get("status", None)
                    actions = messages.get("executor_result", {}).get("actions", None)
                    notification_msg = status + "\n*Action Executed*: " + actions[0].get("_action_", {}).get("name", None) + "\n*Action Description*: " + actions[0].get("_action_", {}).get("description", None)
                    show_hil_action_result(notification_msg)
            elif not hil_payload.get("sel_action", None) or len(hil_payload) == 0:
                hil_placeholder.empty()
                
            st.session_state.chat_input_disabled = False
                        
        human_in_the_loop = {"enabled": False}
        for task in st.session_state.agent_state.tasks:
            if hasattr(task, "interrupts") and isinstance(task.interrupts, tuple) and task.interrupts:
                for interrupt_obj in task.interrupts:
                    if hasattr(interrupt_obj, "value") and isinstance(interrupt_obj.value, dict) and interrupt_obj.value:
                        human_in_the_loop["enabled"] = True
                        for key, value in interrupt_obj.value.items():
                            human_in_the_loop[key] = value
        
        if human_in_the_loop["enabled"]:
            st.session_state.chat_input_disabled = True
            hil_placeholder, hil_form_elements = st.empty(), []
            with hil_placeholder.container():
                with st.form(key='hil_form', clear_on_submit=True):
                    if human_in_the_loop.get("action_prompt", None):
                        st.markdown("<h4>" + human_in_the_loop["action_prompt"] + "</h4>", unsafe_allow_html=True)
                    if human_in_the_loop.get("action_context", None):
                        st.text(human_in_the_loop["action_context"])
                    st.markdown("<br/>", unsafe_allow_html=True)
                    
                    if human_in_the_loop.get("action_controls", None) and isinstance(human_in_the_loop["action_controls"], list):
                        for control in human_in_the_loop["action_controls"]:
                            for key in control:
                                if "actions" in key and isinstance(control[key], list):
                                    hil_form_elements.append("sel_action")
                                    st.pills(
                                        "Please choose an Action from the list to proceed", 
                                        control[key], 
                                        key="sel_action", 
                                        selection_mode="single", 
                                        label_visibility="collapsed", 
                                        help="Select an action to proceed."
                                    )
                                
                                if "text_inputs" in key and isinstance(control[key], list):
                                    for txt_input in control[key]:
                                        hil_form_elements.append(txt_input["name"])
                                        st.text_input(
                                            label=txt_input["display_name"],
                                            key=txt_input["name"],
                                            help=txt_input["description"],
                                            placeholder=txt_input["description"]
                                        )
                                
                                if "choice_inputs" in key and isinstance(control[key], list):
                                    for choice_input in control[key]:
                                        hil_form_elements.append(choice_input["name"])
                                        st.selectbox(
                                            label=choice_input["display_name"],
                                            options=choice_input["options"],
                                            key=choice_input["name"],
                                            help=choice_input["description"],
                                            placeholder=choice_input["description"]
                                        )
                    
                    st.markdown("<br/>", unsafe_allow_html=True)
                    hil_submit_button = st.form_submit_button(
                        label= 'Submit',
                        type="primary",
                        on_click= hil_action,
                        args=(hil_form_elements, self.formatted_chat_history,)
                    )
                        
    def display_airesponse_details(self, _msg_obj):
        def create_charts(charts, datatable):
            charts_plotted = 0
            df_cols = datatable.columns.tolist()
            columns = st.columns(len(charts))
            try:
                for i, chart_data in enumerate(charts):
                    with columns[i]:
                        x_axis_col = chart_data["x_axis_column"] if "x_axis_column" in chart_data else None
                        y_axis_col = chart_data["y_axis_column"] if "y_axis_column" in chart_data else None
                        y_axis_col2 = chart_data["y_axis_column2"] if "y_axis_column2" in chart_data else None
                        cols_list = [x_axis_col, y_axis_col, y_axis_col2] if y_axis_col2 else [x_axis_col, y_axis_col]
                        if all(element in df_cols for element in cols_list):
                            if "bar" in chart_data["chart_type"]:
                                charts_plotted += 1
                                st.bar_chart(
                                    data = datatable,
                                    x = x_axis_col,
                                    y = y_axis_col,
                                    x_label = chart_data["x_axis_label"],
                                    y_label = chart_data["y_axis_label"],
                                    color = y_axis_col2,
                                    stack = True if y_axis_col2 else False
                                )
                            elif "line" in chart_data["chart_type"]:
                                charts_plotted += 1
                                st.line_chart(
                                    data = datatable,
                                    x = x_axis_col,
                                    y = y_axis_col,
                                    x_label = chart_data["x_axis_label"],
                                    y_label = chart_data["y_axis_label"],
                                    color = y_axis_col2
                                )
                            elif "area" in chart_data["chart_type"]:
                                charts_plotted += 1
                                st.area_chart(
                                    data = datatable,
                                    x = x_axis_col,
                                    y = y_axis_col,
                                    x_label = chart_data["x_axis_label"],
                                    y_label = chart_data["y_axis_label"],
                                    color = y_axis_col2
                                ) 
            except Exception as e:
                logging.error(f"Error executing SQL query: {e}", exc_info=True)
            finally:
                if charts_plotted == 0:
                    st.write("No charts to display for the given data.")

        response = Processor._get_key_from_obj(_msg_obj, "response")
        info = Processor._get_key_from_obj(_msg_obj, "info")
        code = Processor._get_key_from_obj(_msg_obj, "code")
        datatable = Processor._get_key_from_obj(_msg_obj, "datatable")
        executor = Processor._get_key_from_obj(_msg_obj, "executor_result")
        charts = Utilities.extract_json_list_of_dicts(response)
        
        tabs = ["RESPONSE", "INFO"]
        if code: tabs.append("CODE")
        if datatable: tabs.append("TABLE")
        if charts and datatable: tabs.append("CHART")
        if executor: tabs.append("TASK")
        tab_objects = st.tabs(tabs)

        for i, tab_name in enumerate(tabs):
            with tab_objects[i]:
                if tab_name == "RESPONSE":
                    st.write(response)
                elif tab_name == "INFO":
                    for key, value in info.items():
                        if "runbook" in key: continue
                        st.write(f"**{key}:** {value}")
                elif tab_name == "CODE" and code:
                    st.code(code, language="sql")
                elif tab_name == "CHART" and charts:
                    create_charts(charts, datatable)
                elif tab_name == "TASK" and executor:
                    st.write(executor)
                elif tab_name == "TABLE" and datatable:
                    csv_file = StringIO(datatable)
                    datatable = pd.read_csv(csv_file, index_col=False)
                    st.dataframe(datatable)

    def _handle_validation_stage(self, is_approved_response=False, print_user_query=True):
        def save_user_feedback(user_query, _agent_response_obj):
            with st.spinner("Saving Feedback..."):
                rating = st.session_state.get("user_rating", None)
                feedback = st.session_state.get("user_feedback", None)
                st.session_state.chat_input_disabled = False
                if rating and int(rating) >= 3: 
                    st.session_state.memory_manager.processor.save_feedback(user_query, _agent_response_obj, rating, feedback)
            st.session_state.memory_manager.clear_session_state_vars(restart=True)

        _agent_response_obj = st.session_state.memory_manager.processor.get_agent_response_obj()
        user_query = st.session_state.memory_manager.processor._get_key_from_obj(_agent_response_obj, "query")
        
        with st.chat_message("assistant"):
            self.display_airesponse_details(_agent_response_obj)
        self._human_in_the_loop()
        st.markdown('<br/>', unsafe_allow_html=True)
        if not is_approved_response:
            feedbk_placeholder = st.empty()
            with feedbk_placeholder.container():
                with st.form(key='feedback_form', clear_on_submit=True):
                    st.markdown('<p style="font-size:10px !important;">Was this response helpful?</p>', unsafe_allow_html=True)
                    user_rating = st.feedback("stars", key="user_rating")
                    user_feedback = st.text_input("Provide additional feedback (optional):", key="user_feedback")
                    st.markdown("<br/>", unsafe_allow_html=True)
                    feedback_submit_button = st.form_submit_button(
                        label= 'Submit',
                        type="primary",
                        on_click= save_user_feedback,
                        args= (user_query, _agent_response_obj)
                    )
        else:
            st.session_state.memory_manager.clear_session_state_vars(restart=False)
        st.markdown('<hr id="chat_divider">', unsafe_allow_html=True)


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()