import re
import dash
from dash import html, dcc, Input, Output
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go
import os
import numpy as np

def read_config(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()
    
# User input
org_file_path = read_config('config.txt') # Get rid of read_config function and replace with 'your_file_path'

# Load the BERT model
model = SentenceTransformer('all-mpnet-base-v2') 

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
def embed_paragraphs(paragraphs, embeddings_file='embeddings.npy'):
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

# Apply and save UMAP
def compute_and_save_umap_embeddings(embeddings, umap_file='umap_embeddings.npy'):
    print("Applying UMAP...")
    umap_reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
    umap_embeddings = umap_reducer.fit_transform(embeddings)
    np.save(umap_file, umap_embeddings)
    return umap_embeddings

# Function to truncate text to a specified length
def truncate(text, length=100):
    return text if len(text) <= length else text[:length] + '...'

# Function to filter entries after a specified year
def filter_entries(cutoff_year, paragraphs, paragraph_details):
    filtered_paragraphs = []
    filtered_details = []
    for i, detail in enumerate(paragraph_details):
        if detail['year'] is not None and int(detail['year']) == int(cutoff_year):  # Check for None and convert year to int for comparison
            filtered_paragraphs.append(paragraphs[i])
            filtered_details.append(detail)
    return filtered_paragraphs, filtered_details

# Function to create a dropdown for year selection
def create_year_dropdown(paragraph_details):
    years = sorted({detail['year'] for detail in paragraph_details if detail['year'] is not None})
    all_years_option = [{'label': 'All Years', 'value': 'all'}]
    year_options = all_years_option + [{'label': year, 'value': year} for year in years]
    return dcc.Dropdown(
        id='year-dropdown',
        options=year_options,
        value='all'  # Default to 'All Years'
    )

# Load and process the data
paragraphs, paragraph_details = read_org_file(org_file_path)

# Compute or load embeddings
all_embeddings = embed_paragraphs(paragraphs)

# Create a mapping from paragraphs to their indices
paragraph_to_index = {p: i for i, p in enumerate(paragraphs)}
np.save('paragraph_to_index.npy', paragraph_to_index)

# Compute or load UMAP embeddings
umap_file = 'umap_embeddings.npy'
if not os.path.exists(umap_file):
    umap_embeddings = compute_and_save_umap_embeddings(all_embeddings, umap_file)
else:
    print("Loading UMAP embeddings from file...")
    umap_embeddings = np.load(umap_file)

# Initialize the Dash app
app = dash.Dash(__name__)

# Create year dropdown
year_dropdown = create_year_dropdown(paragraph_details)

# App layout
app.layout = html.Div([
    html.Div([
        html.Label('Select Year:'),
        year_dropdown
    ]),
    html.Div([
        dcc.Input(id='search-input', type='text', placeholder='Search text...'),
        html.Button('Search', id='search-button'),
    ]),
    dcc.Graph(id='umap-plot'),
    html.Div(id='text-output', style={'white-space': 'pre-wrap', 'word-wrap': 'break-word'})
])

@app.callback(
    Output('umap-plot', 'figure'),
    [Input('search-button', 'n_clicks'), Input('year-dropdown', 'value')],
    [dash.dependencies.State('search-input', 'value')]
)
def update_plot(n_clicks, selected_year, search_value):
    # Load paragraph to index mapping
    if os.path.exists('paragraph_to_index.npy'):
        paragraph_to_index = np.load('paragraph_to_index.npy', allow_pickle=True).item()
    else:
        print("Paragraph to index mapping file not found.")
        return go.Figure()

    # Handle "All Years" option
    if selected_year == 'all':
        filtered_paragraphs = paragraphs
        filtered_details = paragraph_details
    else:
        # Filter paragraphs and details for the selected year
        filtered_paragraphs, filtered_details = filter_entries(selected_year, paragraphs, paragraph_details)

    # Get indices of the filtered paragraphs using the mapping
    filtered_indices = [paragraph_to_index[p] for p in filtered_paragraphs if p in paragraph_to_index]

    # Filter UMAP embeddings based on these indices
    filtered_umap_embeddings = umap_embeddings[filtered_indices]  

    # Create truncated text for tooltips
    truncated_filtered_paragraphs = [truncate(p, length=100) for p in filtered_paragraphs]

    # Highlight search results if search_value is provided
    if search_value:
        matched_indices = [i for i, text in enumerate(filtered_paragraphs) if search_value.lower() in text.lower()]
        marker_colors = ['rgba(200,200,200,0.5)' if i not in matched_indices else 'rgba(0,0,255,1)' for i in range(len(filtered_paragraphs))]
    else:
        marker_colors = ['rgba(0,0,255,1)' for _ in filtered_paragraphs]

    # Generate the UMAP plot with only the filtered data and fixed axis ranges
    return {
        'data': [
            go.Scatter(
                x=filtered_umap_embeddings[:, 0],
                y=filtered_umap_embeddings[:, 1],
                mode='markers',
                text=truncated_filtered_paragraphs,
                hoverinfo='text',
                marker=dict(size=7, color=marker_colors, opacity=0.8)
            )
        ],
        'layout': go.Layout(
            title='Journal Entries Visualization',
            xaxis={'title': 'UMAP Dimension 1', 'range': [2, 14]},
            yaxis={'title': 'UMAP Dimension 2', 'range': [-4, 10]},
            hovermode='closest'
        )
    }


# Callback to update text-output div with full text and date-time details
@app.callback(
    Output('text-output', 'children'),
    [Input('umap-plot', 'clickData'), Input('year-dropdown', 'value')]
)
def display_click_data(clickData, selected_year):
    # Handle "All Years" option
    if selected_year == 'all':
        current_paragraphs = paragraphs
        current_details = paragraph_details
    else:
        # Filter paragraphs and details for the selected year
        current_paragraphs, current_details = filter_entries(selected_year, paragraphs, paragraph_details)

    if clickData is None:
        return 'Click on a point to display its text here.'
    else:
        point_index = clickData['points'][0]['pointIndex']
        if point_index < len(current_paragraphs):
            detail = current_details[point_index]
            full_text = current_paragraphs[point_index]
            date_time_info = f"{detail['year']}, {detail['day']}, {detail['time']}"
            return f"Date: {date_time_info}\nText: {full_text}"
        else:
            return 'No data for selected point.'


# Run the app
app.run_server(debug=True)



