import textwrap

APP_DESCRIPTION = textwrap.dedent("""
    This application is a powerful Q&A chatbot designed to provide comprehensive support for the  (dvt). Built with a robust AI agentic workflow leveraging the REACT framework and the advanced capabilities of the Gemini LLM, this chatbot offers a seamless and intuitive way to access information and perform analytical tasks related to dvt.

    ## Key Capabilities

    * **Comprehensive dvt Knowledge Base:**
        * Answers a wide range of questions related to dvt, drawing from official Confluence documentation and JIRA tickets.
        * Provides in-depth information on:
            * **Platform Functionality:** Understanding features, capabilities, and limitations.
            * **Job Management:** Submitting, monitoring, and debugging batch processing jobs.
            * **Data Processing:** Transforming, analyzing, and managing large datasets (snapshots).
            * **Infrastructure:** Configuring and managing platform resources, components, and services.
            * **Troubleshooting:** Diagnosing and resolving errors, performance issues, and unexpected behavior.
            * **Best Practices:** Implementing efficient and optimized workflows.
            * **Code Examples:** Providing sample code snippets for common tasks.
            * **Configuration:** Guidance on configuring various settings, parameters, and services.
            * **Access Control:** Understanding and managing user permissions and roles.
            * **Cost Optimization:** Strategies for reducing operational costs.
            * **API Usage:** Documentation and examples for interacting with the platform's APIs.
            * **Platform Updates:** Information regarding the latest platform changes and releases.
            * **General Inquiries:** Answering any general question related to the platform.
            * **Error Messages:** Explaining error messages and providing solutions.
            * **Performance Tuning:** Optimizing job performance.
            * **Data Security:** Information regarding data security.
            * **Compliance:** Information regarding platform compliance.
            * **Migration:** Helping with platform migration.
            * **Comprehensive dvt Information Retrieval:** Any question or request for information related to the  (dvt), regardless of specificity or category.
    * **Dynamic Database Analytics:**
        * Generates real-time analytics from the dvt database based on user queries expressed in natural language.
        * Provides on-demand insights and reports, eliminating the need for complex manual data extraction.
    * **Ad-Hoc Dataset Analysis:**
        * Enables users to upload CSV datasets for immediate analysis.
        * Provides instant insights and visualizations, empowering users to explore and understand their data.
    * **Continuous Learning and Improvement:**
        * Collects user feedback for every response.
        * Utilizes feedback to refine and enhance future responses, ensuring accuracy and relevance.
    * **User Interface:**
        * Session Management.

    ## How it Works

    1.  **Natural Language Input:** Users can ask questions or request information in plain English.
    2.  **AI Agentic Workflow:** The application processes the input using an intelligent agentic workflow powered by Gemini LLM.
    3.  **Knowledge Retrieval:** The AI agent retrieves relevant information from dvt Confluence documentation and JIRA tickets.
    4.  **Database Query (if applicable):** If the query involves database analytics, the application generates and executes the necessary SQL queries.
    5.  **CSV Analysis (if applicable):** If the user uploads a CSV, the application parses it and performs the requested analysis.
    6.  **Response Generation:** The AI agent formulates a clear and concise response, including relevant information, code examples, or analytical results.
    7.  **User Feedback:** Users can provide feedback on the response, helping to improve the application's accuracy and effectiveness.

    ## Future Enhancements

    * Integration with additional data sources.
    * Enhanced visualization capabilities for database and CSV analysis.
    * Proactive suggestions and recommendations based on user activity.
    * Improved multi-turn conversation handling.
""")

APP_USER_GUIDE = textwrap.dedent("""
    ## How to Use AskGBP Effectively

    Here are some tips and examples to help you get the most out of the chatbot:

    **1. Clear and Specific Questions:**

    * The more specific your question, the better the chatbot can understand and respond.
    * **Example:** Instead of "My job isn't working," try "Job hasn't started execution for 1.5h after triggering."

    **2. Keyword Optimization:**

    * Use relevant keywords related to dvt terminology.
    * **Example:** "Pipeline Dependency," "DAG," "DataPrep," "conversion plugin," "profile instances," "Header.yml".

    **3. Specifying Data Sources:**

    * You can control the data sources used to answer your questions.
        * **Knowledge Base Only:** Use "Search Knowledge Base Only" at the end of your question to search dvt documentation and JIRA tickets.
            * **Example:** "How to create a custom plugin?. Knowledge Base only."
            * **Example:** "Failed Batch execution doesnt list in airflow. Search Knowledge Base only."
        * **Preapproved Responses Only:** Use "Search Preapproved Responses Only" to retrieve pre-validated responses.
            * **Example:** "Pipeline Dependency not working as expected for Add/Delete and Triggers process. Preapproved Response only."
            * **Example:** "What is Header.yml? Please describe in detail. Preapproved Response only."
            * **Example:** "Describe conversion plugin in detail. Search through Preapproved Response only."
        * **Hybrid Responses Only:** Use "Search Hybrid Responses Only" to search both Knowledge Base and Preapproved responses.
            * **Example:** "How to handle repeated events in dvt to avoid congestion?. Search Hybrid Responses only."
            * **Example:** "Dag is not getting published because of subdag already exists. Hybrid Response only."
    * If you do not specify a source, the app will try its best to provide you with the most relevant data.

    **4. Database Analytics:**

    * You can ask for real-time analytics from the dvt database using natural language queries.
    * **Example:** "Analyze the results for the query: monthly percentage job count split by status from last 12 months from max date sorted by date desc."
    * **Example:** "How many jobs FAILED in 24 hours calculated from the most recent date available?"
    * **Example:** "Analyze the results for the query: daily percentage split of job count across status from last 15 days from max date sorted by date desc."
    * **Example:** "Show list of profile instances of jobs that FAILED in 24 hours calculated from the most recent start date available."
    * **Example:** "show results as table for the query: daily count of COMPLETE jobs with average execution time in minutes from last 15 days from max date sorted by date desc."
    * **Example:** "How many jobs finished with status COMPLETE between September 15, 2024 and September 30, 2024?"

    **5. CSV Data Analysis:**

    * You can upload CSV files for on-the-fly analysis.
    * **Example:** "Analyze the following data in csv format: <SHIFT+ENTER> [CSV data here]"
    * **Example:** "Generate a pivot table with the following structure: Rows: Month (in descending order), Columns: Job Status, Values: Percentage of jobs for each status within each month. The data should cover the last 10 months, calculated from the most recent date available."

    **6. Example Scenarios:**

    * **Troubleshooting:** "dvt sends the same event repeatedly. It is congesting dvt listener since there are too many events."
    * **Error Handling:** "Profile promotion failed from UAT to Prod through config API, getting 500 internal server error."
    * **Congestion Handling:** "How to handle repeated events in dvt to avoid congestion?"

    **7. Iterative Queries and Refinement:**

    * If you don't get the desired result, try rephrasing your question or specifying a data source.
    * **Example:** If the first attempt at "How to handle repeated events in dvt?" doesn't provide the right answer, try "How to handle repeated events in dvt to avoid congestion?. Search Knowledge Base only."

    **8. Providing Feedback:**

    * Your feedback is crucial for improving the chatbot's performance.
    * Please provide feedback on the accuracy and relevance of the responses.
    * If you encounter any issues or have suggestions, please let us know.

    **9. Important Note:**

    * While we strive for accuracy, the application is still under development.
    * If you do not get a response to your query, please rerun the query and try specifying a data source (Knowledge Base or Hybrid).

    By following these guidelines, you can effectively use the dvt AI Chatbot to get the information and analytics you need.                             
""")