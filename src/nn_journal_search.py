import dash
from dash import html, dcc, Input, Output
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def read_config(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()
    
# User input
org_file_path = read_config('data/config.txt') # Get rid of read_config function and replace with 'your_file_path'

# Load the BERT model
model = SentenceTransformer('all-mpnet-base-v2')

# Don't worry about this. 
def read_config(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# Function to read and process the org file
def read_org_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    year, day, time = None, None, None
    paragraphs = []
    paragraph_details = []  # To store details like year, day, and time

    for line in content:
        line = line.strip()
        if line.startswith('**** '):
            time = line.strip('* ')
        elif line.startswith('*** '):
            day = line.strip('* ')
        elif line.startswith('* 20'):
            year = line.strip('* ')
        elif line and not line.isspace():  # Check if line is a non-empty paragraph
            paragraphs.append(line)
            paragraph_details.append({'year': year, 'day': day, 'time': time})
    return paragraphs, paragraph_details

# Embedding paragraphs
def embed_paragraphs(paragraphs, embeddings_file='data/embeddings.npy'):
    # Check if embeddings file exists
    if os.path.exists(embeddings_file):
        print("Loading embeddings from file...")
        # Load embeddings
        embeddings = np.load(embeddings_file, allow_pickle=True)
    else:
        print("Embedding paragraphs...")
        # Compute embeddings
        embeddings = model.encode(paragraphs, show_progress_bar=True)
        # Save embeddings
        np.save(embeddings_file, embeddings)
    return embeddings

# Function to truncate text to a specified length
def truncate(text, length=100):
    return text if len(text) <= length else text[:length] + '...'

# Function to find nearest neighbors
def find_nearest_neighbors(query_embedding, all_embeddings, top_k=10):
    similarities = cosine_similarity(query_embedding, all_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
# App layout
app.layout = html.Div([
    html.Div([
        html.H2("Nearest Neighbor Journal Search", style={'textAlign': 'center'}),
        dcc.Input(
            id='search-input', 
            type='text', 
            placeholder='Enter text to search...',
            style={'width': '50%', 'display': 'inline-block'}
        ),
        html.Button(
            'Search', 
            id='search-button',
            style={'display': 'inline-block', 'marginLeft': '10px'}
        ),
    ], style={'textAlign': 'center', 'padding': '20px 0'}),
    html.Div(id='search-output', style={'white-space': 'pre-wrap', 'word-wrap': 'break-word'})
])

# Load and process the data
paragraphs, paragraph_details = read_org_file(org_file_path)

# Compute or load embeddings
all_embeddings = embed_paragraphs(paragraphs)

@app.callback(
    Output('search-output', 'children'),
    [Input('search-button', 'n_clicks')],
    [dash.dependencies.State('search-input', 'value')]
)
def perform_search(n_clicks, search_value):
    if not search_value:
        return "Enter a query to search."
        
    # Embed the search query
    query_embedding = model.encode([search_value])

    # Find the nearest neighbors
    top_indices = find_nearest_neighbors(query_embedding, all_embeddings)

    # Display the results (without truncating the text)
    results = [f"Date: {paragraph_details[idx]['year']}, {paragraph_details[idx]['day']}, {paragraph_details[idx]['time']}\nText: {paragraphs[idx]}\n" for idx in top_indices]
    return "\n".join(results)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
