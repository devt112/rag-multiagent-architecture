import os, sys, json, textwrap, urllib3, time
from tools import generate_execute_sql_query, generate_data_insights
from agent import NLPSQLAgent
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage


def test_nlpsql_tool(query: str):
    return generate_execute_sql_query.invoke({"query": query}, config={"configurable": {}})

def test_data_insights_tool(user_input: str, sql_query:str, query_result:str):
    return generate_data_insights.invoke({
            "query": user_input,
            "code": sql_query,
            "datatable": query_result
        },
        config={"configurable": {}}
    )

def test_agent(user_inputs: List[str]):
    agent = NLPSQLAgent(thread_id="abc918", debug=True, add_memory_saver=False)
    for user_input in user_inputs:
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print(user_input)
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        result = agent.run_agent(user_input)
        print(len(result["messages"]), "\n----------------------------------------------------------------------------------------------\n", result["messages"][-1].content)
        print("----------------------------------------------------------------------------------------------\n\n")
        time.sleep(5)
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        
def test_graph():
    pass

# if __name__ == "__main__":
user_inputs = [
    "Analyze the results for the query: monthly percentage job count split by status from last 12 months from max date sorted by date desc.",
    # "How many jobs FAILED in 24 hours calculated from the most recent date available?",
    # "Analyze the following data in csv format: execution_month,status,job_count,percentage\r\n2024-09,COMPLETE,13328,87.4196510560146924\r\n2024-09,FAILED,1900,12.4622851895579168\r\n2024-09,RUNNING,18,0.11806375442739079103\r\n2024-08,COMPLETE,9412,81.1099620820406756\r\n2024-08,FAILED,2184,18.8210961737331954\r\n2024-08,RUNNING,8,0.06894174422612892106\r\n2024-07,COMPLETE,6468,70.1366297983083930\r\n2024-07,FAILED,2698,29.2561266536543049\r\n2024-07,RUNNING,56,0.60724354803730210367\r\n2024-06,COMPLETE,2635,64.7261115205109310\r\n2024-06,FAILED,1433,35.2001965119135348\r\n2024-06,RUNNING,3,0.07369196757553426676\r\n2024-05,COMPLETE,1165,62.8032345013477089\r\n2024-05,FAILED,684,36.8733153638814016\r\n2024-05,RUNNING,6,0.32345013477088948787\r\n2024-04,COMPLETE,1036,58.6636466591166478\r\n2024-04,FAILED,678,38.3918459796149490\r\n2024-04,RUNNING,52,2.9445073612684032\r\n2024-03,COMPLETE,1046,62.3361144219308701\r\n2024-03,FAILED,625,37.2467222884386174\r\n2024-03,RUNNING,7,0.41716328963051251490\r\n2024-02,COMPLETE,696,61.5929203539823009\r\n2024-02,FAILED,426,37.6991150442477876\r\n2024-02,RUNNING,8,0.70796460176991150442\r\n2024-01,COMPLETE,398,55.3546592489568846\r\n2024-01,FAILED,244,33.9360222531293463\r\n2024-01,RUNNING,77,10.7093184979137691\r\n2023-12,COMPLETE,721,68.6666666666666667\r\n2023-12,FAILED,324,30.8571428571428571\r\n2023-12,RUNNING,5,0.47619047619047619048\r\n2023-11,COMPLETE,489,66.8946648426812585\r\n2023-11,FAILED,241,32.9685362517099863\r\n2023-11,RUNNING,1,0.13679890560875512996\r\n2023-10,COMPLETE,467,65.1324965132496513\r\n2023-10,FAILED,250,34.8675034867503487\r\n2023-09,COMPLETE,86,77.4774774774774775\r\n2023-09,FAILED,25,22.5225225225225225",
    # "Analyze the results for the query: daily percentage split of job count across status from last 15 days from max date sorted by date desc.",
    # "Show list of profile instances of jobs that FAILED in 24 hours calculated from the most recent start date available.",
    # "Generate a pivot table with the following structure: Rows: Month (in descending order), Columns: Job Status, Values: Percentage of jobs for each status within each month. The data should cover the last 10 months, calculated from the most recent date available.",
    # "Analyze results as table for the query: daily count of COMPLETE jobs with average execution time in minutes from last 15 days from max date sorted by date desc."
]

os.system('cls' if os.name == 'nt' else 'clear')
# for i in range(10):
# test_agent(user_inputs)
# time.sleep(5)
    

result = test_nlpsql_tool(user_inputs[0])
print(result)
# print("====================================================================================")
# data_insights_result = test_data_insights_tool(result["datatable"], result["code"], result["datatable"])
# print(data_insights_result["response"])


# user_inputs = [
#     "Analyze the following data in csv format: execution_month,status,job_count,percentage\r\n2024-09,COMPLETE,13328,87.4196510560146924\r\n2024-09,FAILED,1900,12.4622851895579168\r\n2024-09,RUNNING,18,0.11806375442739079103\r\n2024-08,COMPLETE,9412,81.1099620820406756\r\n2024-08,FAILED,2184,18.8210961737331954\r\n2024-08,RUNNING,8,0.06894174422612892106\r\n2024-07,COMPLETE,6468,70.1366297983083930\r\n2024-07,FAILED,2698,29.2561266536543049\r\n2024-07,RUNNING,56,0.60724354803730210367\r\n2024-06,COMPLETE,2635,64.7261115205109310\r\n2024-06,FAILED,1433,35.2001965119135348\r\n2024-06,RUNNING,3,0.07369196757553426676\r\n2024-05,COMPLETE,1165,62.8032345013477089\r\n2024-05,FAILED,684,36.8733153638814016\r\n2024-05,RUNNING,6,0.32345013477088948787\r\n2024-04,COMPLETE,1036,58.6636466591166478\r\n2024-04,FAILED,678,38.3918459796149490\r\n2024-04,RUNNING,52,2.9445073612684032\r\n2024-03,COMPLETE,1046,62.3361144219308701\r\n2024-03,FAILED,625,37.2467222884386174\r\n2024-03,RUNNING,7,0.41716328963051251490\r\n2024-02,COMPLETE,696,61.5929203539823009\r\n2024-02,FAILED,426,37.6991150442477876\r\n2024-02,RUNNING,8,0.70796460176991150442\r\n2024-01,COMPLETE,398,55.3546592489568846\r\n2024-01,FAILED,244,33.9360222531293463\r\n2024-01,RUNNING,77,10.7093184979137691\r\n2023-12,COMPLETE,721,68.6666666666666667\r\n2023-12,FAILED,324,30.8571428571428571\r\n2023-12,RUNNING,5,0.47619047619047619048\r\n2023-11,COMPLETE,489,66.8946648426812585\r\n2023-11,FAILED,241,32.9685362517099863\r\n2023-11,RUNNING,1,0.13679890560875512996\r\n2023-10,COMPLETE,467,65.1324965132496513\r\n2023-10,FAILED,250,34.8675034867503487\r\n2023-09,COMPLETE,86,77.4774774774774775\r\n2023-09,FAILED,25,22.5225225225225225"
# ]
# test_agent(user_inputs)

# result = test_data_insights_tool(user_inputs[0], "", "")
# print(result["data_analysis"])
