import spacy
import os, json, csv
from bs4 import BeautifulSoup
from collections import Counter

def extract_and_rank_entities_recursive(folder_path):
    """
    Extracts and ranks all named entities from all .txt files in a given folder
    and its subdirectories, sorted by frequency in ascending order.

    Args:
        folder_path (str): The path to the folder containing the text documents.

    Returns:
        list: A list of all entities and their frequencies, sorted by frequency (ascending).
    """
    nlp = spacy.load("en_core_web_sm")
    all_entities = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "content" in data:
                            html_text = data["content"]
                            soup = BeautifulSoup(html_text, 'html.parser')
                            text = soup.get_text(separator=' ')  # Extract text, separating with spaces
                            doc = nlp(text)
                            for ent in doc.ents:
                                all_entities.append(ent.text.lower())
                except Exception as e:
                    print(f"Error reading or processing {filename}: {e}")

    entity_counts = Counter(all_entities)
    ranked_entities_ascending = sorted(entity_counts.items(), key=lambda item: item[1])

    return ranked_entities_ascending

if __name__ == "__main__":
    folder_path = r"C:\dev\confluence-htmls"
    ranked_entities = extract_and_rank_entities_recursive(folder_path)

    print(f"All entities ranked by frequency (ascending order) across all documents and subdirectories:")
    for entity, count in ranked_entities:
        print(f"- {entity}: {count}")
        
    with open(r"C:\dev\99999_DVT_ASKGBP_AI_MULTIAGENT\retrieval-engine\entities.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Entity', 'Frequency'])  # Write header row
        writer.writerows(ranked_entities)