# Journal Mapper

This project takes a user's journal and creates a spatial embedding out of it, that can be mapped and searched.

## Description

A more in-depth writeup of this project can be found [here](https://tjburns08.github.io/tech_enabled_journaling.html). It assumes that the journal or notes are in a plain text (org mode) format. The scripts read in the file, produce a BERT embedding (via all-mpnet-base-v2 in the sentence-transformers package), and perform UMAP on the embedding. journal_map.py provides a web interface to display a searchable umap of your jounral entries, where each point is a paragraph and similar paragraphs by context are grouped near each other. nn_journal_search.py provides a web interface that allows for a nearest neighbor search of a given piece of text to the nearest journal pargarphs by context.

## Getting Started

To use the scripts as they are, the journal must be an org mode file (this can be modified to be plain text or any other format if you don't use org mode). The file must have the following bullet point format:

`* Year`
`*** Date`
`**** Time`
Journal entry goes here.

To make this explicit:

`* 2023`
`*** July 4`
`**** 10:22pm`
Journal entry paragraph 1...

Journal entry paragraph 2...

The time does not need to be of any particular format. Any string will do. For a given entry, paragraphs must be individual lines, separated by an empty line, as shown above.

### Dependencies

I wrote the scripts using Python 3.11.4. The install requirements can be found in requirements.txt.

### Executing program

Run the scripts as they are. They only thing that you need to change is the directory to your journal file. This can be found at the top of each of the scripts, labeled org_file_path.

When you run the scripts for the first time, embeddings for the journal will be created (embeddings.npy) along with a UMAP embedding (umap_embeddings.npy) and a paragraph indexing file (paragraph_to_index.npy). When you run the scripts subsequent times, they will look for these files and upload them, so you don't have to redo everything.

In the event that you change the journal file in any way, please delete these saved files and re-run everything. If you don't, the indexing will probably be messed up (you'll see it in the umap, as the search terms will appear to be randomly distributed through the map).

