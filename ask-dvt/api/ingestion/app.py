import os, re, warnings, argparse, concurrent.futures, time, psycopg2, urllib.parse
from bs4 import BeautifulSoup
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from pgvector.psycopg2 import register_vector
from flask import Flask, request
from flask_cors import CORS
from atlassian import Confluence
from jira.client import JIRA


app = Flask(__name__)
CORS(app)

PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
DB_HOST = "host.docker.internal"
# DB_HOST = "localhost"
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_PORT = os.environ.get("DB_PORT")

CONFLUENCE_URL = "https://.atlassian.net/wiki"
USERNAME = os.environ.get("CONFLUENCE_USERNAME")
API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN")

jira_server = 'https://.atlassian.net'
username = os.environ.get("JIRA_USERNAME")
api_token = os.environ.get("JIRA_API_TOKEN")

aiplatform.init(project=PROJECT_ID, location=REGION)

def create_connection():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    register_vector(conn)
    return conn


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    length_function=len,
    is_separator_regex=False,
    chunk_overlap=50,
    separators=["#HEADER#", "#SUBHEADER#", "#SUBHEADER#", " "]
)


def chunk_text(text):
    text = text.replace("<h1>", "#HEADER#").replace("<h2>", "#SUBHEADER#").replace("<strong>", "#SUBHEADER#")
    soup = BeautifulSoup(text, "html.parser")
    for data in soup(['style', 'script']):
        data.decompose()
    html = ' '.join(soup.stripped_strings)
    return text_splitter.split_text(html)


def embed_text(text):
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    embeddings = model.get_embeddings([text])
    return np.array(embeddings[0].values)  # Return as numpy array for pgvector


def check_if_record_exists(space, pageid):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM embeddings e
            JOIN confluence_pages cnfl ON e.source_id = cnfl.id
            JOIN sources s ON s.id = e.source
            WHERE cnfl.space = %s AND cnfl.pageid = %s
        );
    """, (space, pageid))
    exists = cur.fetchone()
    if exists == False: print(exists)
    cur.close()
    conn.close()
    return exists


def add_confluence_documents_to_db(documents, params):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO confluence_pages (space, pageid)
        VALUES (%s, %s)
        ON CONFLICT (space, pageid) DO UPDATE SET space = EXCLUDED.space, pageid=excluded.pageid
        RETURNING id;
    """, (params["space"], params["pageid"]))
    confluence_page_id = cur.fetchone()

    for i, doc in enumerate(documents):
        embedding = embed_text(doc)
        cur.execute("""
            INSERT INTO embeddings (source, source_id, chunk_id, chunk_text, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source, source_id, chunk_id) DO UPDATE
            SET chunk_text = excluded.chunk_text, embedding = excluded.embedding;
        """, (1, confluence_page_id, i + 1, doc, embedding))

    conn.commit()
    cur.close()
    conn.close()      


def add_jira_documents_to_db(documents, params):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO jira_issues (issueid, issuekey)
        VALUES (%s, %s)
        ON CONFLICT (issueid, issuekey) DO UPDATE SET issueid = EXCLUDED.issueid, issuekey=excluded.issuekey
        RETURNING id;
    """, (params["issueid"], params["key"]))
    jira_issue_id = cur.fetchone()

    for i, doc in enumerate(documents):
        embedding = embed_text(doc)
        cur.execute("""
            INSERT INTO embeddings (source, source_id, chunk_id, chunk_text, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source, source_id, chunk_id) DO UPDATE
            SET chunk_text = excluded.chunk_text, embedding = excluded.embedding;
        """, (2, jira_issue_id, i + 1, doc, embedding))
        
    conn.commit()
    cur.close()
    conn.close() 
    

def get_confluence_page(page_id):
    confluence = Confluence(
        url=CONFLUENCE_URL,
        username=USERNAME,
        password=API_TOKEN,
        verify_ssl=False
    )
    spacekey = confluence.get_page_space(page_id)
    page_data = confluence.get_page_by_id(page_id, expand='body.storage', status=None, version=None) if page_id is not None else ""            
    if len(page_data) > 0:           
        title = page_data["title"]        
        page_content = page_data["body"]["storage"]["value"]
        soup = BeautifulSoup(page_content, 'html.parser')
        if len(soup.get_text(strip=True).strip()) > 0:
            page_content = title + "<br> " + page_content
            return {"space": spacekey, "page_content":page_content} 
    return {"space": spacekey, "page_content":page_data}


def get_confluence_pageid(url):
    try:
        parsed_url = urllib.parse.urlparse(url)
        path_parts = parsed_url.path.split('/')
        for i, part in enumerate(path_parts):
            if part == "pages" and i + 1 < len(path_parts):
              try:
                page_id = int(path_parts[i + 1])
                return str(page_id)
              except ValueError:
                return None
        return None
    except Exception as e:
        print(f"Error parsing URL: {e}")
        return None   


def create_confluence_embeddings(content, params):
    documents = chunk_text(content)
    status = add_confluence_documents_to_db(documents, params)
    return f"{len(documents)} chunks created."


def create_jira_embeddings(jdata):
    def prep_content(jdata, key, tag):
        if key in jdata and jdata[key] is not None and len(jdata[key]) > 0:
            return tag + " " + jdata[key]
        else: return ""

    category = prep_content(jdata, "category", "Type of issue is")
    root_cause = prep_content(jdata, "root_cause", "Root cause of the reported issue is")
    resolution = prep_content(jdata, "resolution", "Details of how the issue was resolved or resolution notes")
    summary = prep_content(jdata, "summary", "Summary or problem statement or user query is")
    description = prep_content(jdata, "description", "Details of the issue:")
    comments = " ".join(jdata["comments"]).strip()
    content = " ".join([summary, description, root_cause, resolution, category, comments])
    documents = chunk_text(content)
    add_jira_documents_to_db(documents, jdata)
    return f"{len(documents)} chunks created."


class CustomFieldOption:
    def __init__(self, id, value):
        self.id = id
        self.value = value

     
def custom_field_option_to_dict(option):
    if option is None:
        return {'id': None, 'value': ""}
    return {'id': option.id, 'value': option.value}


def get_jira_page(jiraid):
    jira_options={'server': jira_server, 'verify': False}
    jira = JIRA(options=jira_options, basic_auth=(username, api_token)) 
    issue = jira.issue(jiraid)
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
    return jdata
    

@app.route("/vectorize/confluence/page", methods=["GET", "POST"])
def vectorize_confluence_page():
    if request.method == "POST":
        data = request.get_json()
        if data and "url" in data:
            page_url = data["url"]
            page_id = get_confluence_pageid(page_url)
            page = get_confluence_page(page_id)            
            page = create_confluence_embeddings(page["page_content"], {"space": page["space"], "pageid": page_id}) if len(page["page_content"]) > 0 else f"0 chunks created."
            return f"{page}" , 200 #200 OK
        else:
            return "Invalid POST request. Please provide a 'url' in JSON.", 400 #400 Bad Request
    else:
        return "Hello from GET!" , 200


@app.route("/vectorize/jira/page", methods=["GET", "POST"])
def vectorize_jira_page():
    if request.method == "POST":
        data = request.get_json()
        if data and "jiraid" in data:
            page_content = get_jira_page(data["jiraid"])            
            return f"{create_jira_embeddings(page_content)}" , 200 #200 OK
        else:
            return "Invalid POST request. Please provide a 'url' in JSON.", 400 #400 Bad Request
    else:
        return "Hello from GET!" , 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) # Important: host="0.0.0.0"