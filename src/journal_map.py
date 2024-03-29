import re
import dash
from dash import html, dcc, Input, Output
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

def read_config(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

def select_org_file(org_files):
    for i, file in enumerate(org_files):
        print(f"{i + 1}: {file}")
    choice = int(input("Select an org file by number: ")) - 1
    return org_files[choice]

org_files = read_config('data/config.txt')
org_file_path = select_org_file(org_files)
print(f"Selected Org File: {org_file_path}")
    
# User input
# org_file_path = read_config('data/config.txt') # Get rid of read_config function and replace with 'your_file_path'

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
        if line.startswith("#+") or line.startswith("[["):
            continue
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

# Calculate a KNN density estimate of the embeddings
def compute_knn_density(embeddings, k=200, epsilon=1e-5):
    # Create a k-NN model and fit it
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(embeddings)

    # Find distances and indices of k-neighborhood
    distances, indices = nn.kneighbors(embeddings)

    # Compute density (you can modify this calculation as needed)
    density = 1 / (np.mean(distances, axis=1) + epsilon)

    # Get rid of extremes
    lower_bound, upper_bound = np.percentile(density, [5, 90])
    density = np.clip(density, lower_bound, upper_bound)

    # Normalize density for coloring
    normalized_density = (density - np.min(density)) / (np.max(density) - np.min(density))

    return normalized_density


# Apply and save UMAP
def compute_and_save_umap_embeddings(embeddings, umap_file='data/umap_embeddings.npy'):
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
np.save('data/paragraph_to_index.npy', paragraph_to_index)

# Compute or load UMAP embeddings
umap_file = 'data/umap_embeddings.npy'
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
    html.Div([
        dcc.Checklist(
            id='density-checklist',
            options=[
                {'label': 'Color by kNN Density', 'value': 'KNN'}
            ],
            value=[]
        ),
    ]),
    dcc.Graph(id='umap-plot'),
    html.Div(id='text-output', style={'white-space': 'pre-wrap', 'word-wrap': 'break-word'}),
])

@app.callback(
    Output('umap-plot', 'figure'),
    [Input('search-button', 'n_clicks'), 
     Input('year-dropdown', 'value'), 
     Input('density-checklist', 'value')],
    [dash.dependencies.State('search-input', 'value')]
)
def update_plot(n_clicks, selected_year, checklist_values, search_value):
    # Load paragraph to index mapping
    if os.path.exists('data/paragraph_to_index.npy'):
        paragraph_to_index = np.load('data/paragraph_to_index.npy', allow_pickle=True).item()
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

    # Base color determined by search presence
    if search_value:
        matched_indices = [i for i, text in enumerate(filtered_paragraphs) if search_value.lower() in text.lower()]
        base_colors = ['rgba(200,200,200,0.2)' if i not in matched_indices else 'rgba(0,0,255,1)' for i in range(len(filtered_paragraphs))]
    else:
        base_colors = ['rgba(0,0,255,1)' for _ in filtered_paragraphs]

    # Apply kNN density-based coloring if checkbox is checked
    color_by_density = "KNN" in checklist_values
    if color_by_density:
        # Load or compute kNN density, then apply coloring
        density_file = 'data/density.npy'
        if os.path.exists(density_file):
            print("Loading density values from file...")
            knn_density = np.load(density_file)
        else:
            print("Computing kNN density values...")
            knn_density = compute_knn_density(all_embeddings)
            np.save(density_file, knn_density)

        color_scale = px.colors.sequential.Plasma
        density_colors = [color_scale[int(value * (len(color_scale) - 1))] for value in knn_density[filtered_indices]]
        marker_colors = density_colors
    else:
        marker_colors = base_colors
    
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
            # xaxis={'title': 'UMAP Dimension 1', 'range': [2, 14]},
            # yaxis={'title': 'UMAP Dimension 2', 'range': [-4, 10]},
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
if __name__ == '__main__':
    app.run_server(debug=False)



