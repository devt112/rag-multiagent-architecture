import numpy as np
from datetime import datetime
import re, os, sys, json, logging, vertexai, time
ask_gbp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ask-dvt'))
sys.path.append(ask_gbp_path)
from db_helper import pgdbWrapper
from atlassian import Confluence
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

def get_elapsed_days(datestring):
  today = datetime.today()
  past_datetime_utc = datetime.fromisoformat(datestring.replace('Z', '+00:00'))
  today_utc = datetime.utcnow()
  
  if today.tzinfo is None or today.tzinfo.utcoffset(today) is None:
      import pytz
      utc_timezone = pytz.utc
      today_utc = utc_timezone.localize(today)

  time_difference = today_utc - past_datetime_utc
  elapsed_days = time_difference.days
  return elapsed_days

class ConfluenceEngine:
    def __init__(self, download_folder=None):
        self.db = pgdbWrapper()
        self.download_folder = download_folder
        CONFLUENCE_URL = "https://.atlassian.net/wiki"
        CONFLUENCE_USER = os.environ.get("CONFLUENCE_USER")
        CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN")
        self.confluence = Confluence(
            url=CONFLUENCE_URL,
            username=CONFLUENCE_USER,
            password=CONFLUENCE_API_TOKEN,
            verify_ssl=False
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            length_function=len,
            is_separator_regex=False,
            chunk_overlap=200,
            separators=["#HEADER#", "#SUBHEADER#", "#SUBHEADER#", " "]
        )
        
    def get_page_content(self, page):
        pageid = str(page["id"])
        title = page["title"]
        page_content = page["body"]["storage"]["value"]
        expandable = page["_expandable"]
        SPACE_KEY = expandable["space"].split('/')[-1]
        version = page["version"]["number"]
           
        jdata = {
            "url": page["_links"]["self"],
            "space": SPACE_KEY,
            "pageid": pageid,
            "version": version,
            "title": title,
            "content": page_content
        }
        
        return jdata    

    def download_all_pages_for_space(self, space_keys):
        for SPACE_KEY in space_keys:
            print("Downloading pages for space: ", SPACE_KEY)
            space_folder = os.path.join(self.download_folder, SPACE_KEY)
            os.makedirs(space_folder, exist_ok=True)
            
            s, l, flag, arrPages = 0, 100, True, []

            while flag:
                pages = self.confluence.get_space_content(
                    SPACE_KEY, depth="all", start=s, limit=l, content_type='page', expand='body.storage,history.lastUpdated,metadata.labels,version'
                )
                
                if len(pages["results"]) == 0:
                    break
                else:
                    s = s + l
                    
                for result in pages["results"]:
                    # if get_elapsed_days(result["history"]["lastUpdated"]["when"]) > 2:
                    #     continue
                    
                    jdata = self.get_page_content(result)
                    pageid = jdata["pageid"]
                    soup = BeautifulSoup(jdata["content"], 'html.parser')
                    
                    if pageid not in arrPages and len(soup.get_text(strip=True).strip()) > 0:
                        arrPages.append(pageid)
                        file_path = os.path.join(space_folder, str(pageid) + ".json")
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(jdata, f, ensure_ascii=False, indent=4)

            query = """
                INSERT INTO confluence_pages (space, pageid)
                VALUES (%s, %s)
                ON CONFLICT (space, pageid)
                DO NOTHING;
            """
            data_to_insert = [(SPACE_KEY, page) for page in arrPages]
            success =self.db.execute_bulk_insert(query, data_to_insert)
            if success: print(f"Total Pages for {SPACE_KEY}: {len(arrPages)}")

    def get_child_pages_recursively(self, page_id, child_type='page', processed_pages=None):
        if processed_pages is None:
            processed_pages = set()

        all_child_ids = []
        start = 0
        limit = 100
        SPACE_KEY = None

        while True:
            try:
                pages = self.confluence.get_page_child_by_type(
                    page_id, type=child_type, start=start, limit=limit, expand='body.storage,history.lastUpdated,metadata.labels,version'
                )
            except Exception as e:
                print(f"Error retrieving child pages for page ID {page_id}: {e}")
                break

            if not pages:
                break
            else:
                for page in pages:
                    # if get_elapsed_days(page["history"]["lastUpdated"]["when"]) > 2:
                    #     continue
                    
                    jdata = self.get_page_content(page)
                    page_id_current = jdata["pageid"]
                    SPACE_KEY = jdata["space"]
                        
                    if page_id_current not in processed_pages:
                        processed_pages.add(page_id_current)
                        all_child_ids.append(page_id_current)
                        
                        soup = BeautifulSoup(jdata["content"], 'html.parser')
                        
                        if len(soup.get_text(strip=True).strip()) > 0:
                            os.makedirs(os.path.join(self.download_folder, SPACE_KEY), exist_ok=True)
                            file_path = os.path.join(self.download_folder, SPACE_KEY, f"{page_id_current}.json")
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(jdata, f, ensure_ascii=False, indent=4)
                            # print(page["id"], page["title"]) # Optional logging

                        descendant_ids = self.get_child_pages_recursively(
                            page_id_current, child_type, processed_pages
                        )
                        all_child_ids.extend(descendant_ids)

                start += limit
                
        query = """
            INSERT INTO confluence_pages (space, pageid)
            VALUES (%s, %s)
            ON CONFLICT (space, pageid)
            DO NOTHING;
        """
        data_to_insert = [(SPACE_KEY, page) for page in all_child_ids]
        success = self.db.execute_bulk_insert(query, data_to_insert)
        if success: print(f"Total Pages for {SPACE_KEY}: {len(all_child_ids)}")
        
        return all_child_ids

    def add_confluence_documents_to_db(self, documents, params):
        def embed_text(text):
            model = TextEmbeddingModel.from_pretrained("text-embedding-005")
            embeddings = model.get_embeddings([text])
            return np.array(embeddings[0].values) 

        data_to_insert = []
        _space = params["space"]
        _pageid = params["pageid"]
        for i, doc in enumerate(documents):
            doc = doc.replace("#HEADER#", "").replace("#SUBHEADER#", "").replace("#SUBHEADER#", "")
            if len(doc.strip()) == 0:
                continue
            embedding = embed_text(doc)  # Assuming embed_text is defined elsewhere
            data_to_insert.append((1, _space, _pageid, i + 1, doc, embedding))

        query = """
            INSERT INTO embeddings (source, source_id, chunk_id, chunk_text, embedding)
            VALUES (%s, (SELECT id FROM confluence_pages WHERE space = %s AND pageid = %s), %s, %s, %s)
            ON CONFLICT (source, source_id, chunk_id) DO UPDATE
            SET chunk_text = excluded.chunk_text, embedding = excluded.embedding;
        """
        success = self.db.execute_bulk_insert(query, data_to_insert)
        if not success: print("Bulk insert of embeddings failed!")
        else: print(f"Successfully inserted {len(data_to_insert)} embeddings.")

    def chunk_text(self, text):
        text = text.replace("<h1>", "#HEADER#").replace("<h2>", "#SUBHEADER#").replace("<strong>", "#SUBHEADER#")
        soup = BeautifulSoup(text, "html.parser")
        for data in soup(['style', 'script']):
            data.decompose()
        html = ' '.join(soup.stripped_strings)
        return self.text_splitter.split_text(html)
    
    def create_embeddings(self, filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            jdata = json.load(file)
            documents = self.chunk_text("Link to Reference Document: " + jdata["url"] + ". Context: " + jdata["title"] + ". Details are as follows: " + jdata["content"])
            self.add_confluence_documents_to_db(documents, {"space": jdata["space"], "pageid": jdata["pageid"]})
        return f"Processed: {filepath}"

    def get_missing_pageids(self, pageids):
        query = """
            WITH cte AS (
                SELECT pageid
                FROM UNNEST(%s) AS pageid
            )
            SELECT pageid FROM cte
            WHERE pageid NOT IN (
                SELECT DISTINCT cnfl.pageid
                FROM embeddings e
                JOIN confluence_pages cnfl ON e.source_id = cnfl.id
                JOIN sources s ON s.id = e.source
                where e.source=1
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
        
        # for filename in file_paths:
        #     if filename.endswith(".json"):
        #         filepath = os.path.join(directory, filename)
        #         self.create_embeddings(filepath)
        # else:
        pool_size, start_time, pageids = 10, time.time(), []
        # pageids = [os.path.splitext(os.path.basename(filename))[0] for filename in file_paths if filename.endswith(".json")]
        # missing_pageids = [mpgid[0] for mpgid in self.get_missing_pageids(pageids)]
        # print(len(pageids), len(missing_pageids))

        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = []
            for filename in file_paths:
                filepath = os.path.join(directory, filename)
                futures.append(executor.submit(self.create_embeddings, filepath))

            for future in concurrent.futures.as_completed(futures):
                    result = future.result()    
    

c = ConfluenceEngine(download_folder=r"C:\dev\confluence-htmls")
spaces = ["ABGT", "APPENG", "CB", "GBPUD"]
c.download_all_pages_for_space(spaces)
c.get_child_pages_recursively("614900012", child_type='page') # use "attachment" to download images

# c.create_embeddings(r"C:\dev\confluence-htmls\APPENG\572954879.json")
c.vectorize_documents(r"C:\dev\confluence-htmls")
