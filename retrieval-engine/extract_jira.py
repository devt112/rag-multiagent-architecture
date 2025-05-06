import numpy as np
from jira.client import JIRA
import re, os, sys, json, logging, vertexai, time
ask_gbp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ask-dvt'))
sys.path.append(ask_gbp_path)
from db_helper import pgdbWrapper
import os, warnings, json, concurrent.futures
from bs4 import BeautifulSoup
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
REGION = os.environ.get("GOOGLE_CLOUD_REGION")
print(PROJECT_ID, REGION)
print("******************************************")
vertexai.init(project=PROJECT_ID, location=REGION)

class CustomFieldOption:
    def __init__(self, id, value):
        self.id = id
        self.value = value
        

class JIRAEngine:
    def __init__(self):
        self.db = pgdbWrapper()
        jira_server = 'https://.atlassian.net'
        jira_options={'server': jira_server, 'verify': False}
        username = os.environ.get("CONFLUENCE_USER")
        api_token = os.environ.get("CONFLUENCE_API_TOKEN")
        self.jira = JIRA(options=jira_options, basic_auth=(username, api_token)) 
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            length_function=len,
            is_separator_regex=False,
            chunk_overlap=250,
            separators=["\n\n", " "]
        )

    def download_all_jira_issues(self, filters, root_dir):
        def custom_field_option_to_dict(option):
            if option is None:
                return {'id': None, 'value': ""}
            return {'id': option.id, 'value': option.value}

        for filter_id in filters:
            start_at, max_results, all_issues = 0, 100, []

            while True:
                issues = self.jira.search_issues(f"filter = {filter_id}", startAt=start_at, maxResults=max_results)
                for issue in issues:
                    if issue.id not in [item['id'] for item in all_issues]:
                        all_issues.append({"id": issue.id, "key": issue.key})
                    comments = []
                    for comment in issue.fields.comment.comments:
                        if "Automation for Jira" not in comment.author.displayName:
                            comment_body = re.sub(r"\[~accountid:[^\]]+\]", "", comment.body) 
                            comments.append(comment_body)
                    jdata = {
                        "issueid": issue.id,
                        "key": issue.key,
                        "category": custom_field_option_to_dict(issue.fields.customfield_10702)["value"],
                        "environment": custom_field_option_to_dict(issue.fields.customfield_10703)["value"],
                        "root_cause": custom_field_option_to_dict(issue.fields.customfield_10704)["value"],
                        "resolution": issue.fields.customfield_11501,
                        "summary": issue.fields.summary,
                        "description": issue.fields.description,
                        "comments": comments
                    }

                    with open(os.path.join(root_dir, str(issue.id) + ".json"), 'w', encoding='utf-8') as f:
                        json.dump(jdata, f, ensure_ascii=False, indent=4)

                if len(issues) < max_results:
                    break

                start_at += max_results

            query = """
                INSERT INTO jira_issues 
                (issueid, issuekey) VALUES (%s, %s)
                ON CONFLICT (issueid, issuekey)
                DO NOTHING;
            """
            data_to_insert = [(issue["id"], issue["key"]) for issue in all_issues]
            success = self.db.execute_bulk_insert(query, data_to_insert)
            if success: print(f"Total Pages for {filter_id}: {len(all_issues)}")
            
        if self.jira:
            self.jira.close()

    def add_jira_documents_to_db(self, documents, params):
        def embed_text(text):
            model = TextEmbeddingModel.from_pretrained("text-embedding-005")
            embeddings = model.get_embeddings([text])
            return np.array(embeddings[0].values) 

        data_to_insert = []
        _issueid = params["issueid"]
        _issuekey = params["key"]
        
        for i, doc in enumerate(documents):
            if len(doc.strip()) == 0:
                continue
            embedding = embed_text(doc)
            data_to_insert.append((2, _issueid, _issuekey, i + 1, doc, embedding))

        query = """
            INSERT INTO embeddings (source, source_id, chunk_id, chunk_text, embedding)
            VALUES (%s, (SELECT id FROM jira_issues WHERE issueid = %s AND issuekey = %s), %s, %s, %s)
            ON CONFLICT (source, source_id, chunk_id) DO UPDATE
            SET chunk_text = excluded.chunk_text, embedding = excluded.embedding;
        """
        success = self.db.execute_bulk_insert(query, data_to_insert)
        if not success: print("Bulk insert of embeddings failed!")
        else: print(f"Successfully inserted {len(data_to_insert)} embeddings.")

    def chunk_text(self, text):
        return self.text_splitter.split_text(text)
    
    def create_embeddings(self, filepath):
        def prep_content(jdata, key, tag):
            if key in jdata and jdata[key] is not None and len(jdata[key]) > 0:
                return tag + " " + jdata[key]
            else: return ""
            
        with open(filepath, "r", encoding="utf-8") as file:
            jdata = json.load(file)
            category = prep_content(jdata, "category", "Type of issue is")
            root_cause = prep_content(jdata, "root_cause", "Root cause of the reported issue is")
            resolution = prep_content(jdata, "resolution", "Details of how the issue was resolved or resolution notes are")
            summary = prep_content(jdata, "summary", "Summary or problem statement or user query is")
            description = prep_content(jdata, "description", "Details of the issue:")
            comments = " ".join(jdata["comments"]).strip()
            content = "\n\n".join(["Link to JIRA issue: https://.atlassian.net/browse/" + jdata["key"], summary, description, root_cause, resolution, category, comments])
            documents = self.chunk_text(content)
            self.add_jira_documents_to_db(documents, jdata)
            
        return f"Processed: {filepath}"

    def get_missing_pageids(self, pageids):
        query = """
            WITH cte AS (
                SELECT pageid
                FROM UNNEST(%s) AS pageid
            )
            SELECT pageid FROM cte
            WHERE pageid NOT IN (
                select distinct j.issueid
                from embeddings e
                join jira_issues j on e.source_id=j.id
                join sources s on s.id=e.source
                where e.source=2
            );
        """
        params = (pageids,)
        missing_pageids = self.db.execute_select_query(query, params)
        return missing_pageids

    def vectorize_documents(self, directory):    
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)    

        pool_size, start_time, pageids = 10, time.time(), []
        pageids = [os.path.splitext(filename)[0] for filename in file_paths if filename.endswith(".json")]
        missing_pageids = [mpgid[0] for mpgid in self.get_missing_pageids(pageids)]
        print(len(pageids), len(missing_pageids))
        
        # for filename in file_paths:
        #     if filename.endswith(".json") and os.path.splitext(filename)[0] in missing_pageids:
        #         filepath = os.path.join(directory, filename)
        #         self.create_embeddings(filepath)

        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = []
            for filename in file_paths:
                if filename.endswith(".json") and os.path.splitext(filename)[0] in missing_pageids:
                    filepath = os.path.join(directory, filename)
                    futures.append(executor.submit(self.create_embeddings, filepath))

            for future in concurrent.futures.as_completed(futures):
                    result = future.result()  
    

j = JIRAEngine()
j.download_all_jira_issues(["153027"], r"C:\dev\jira-pages") #["152795","152796"]  "152795","152796"
j.vectorize_documents(r"C:\dev\jira-pages")