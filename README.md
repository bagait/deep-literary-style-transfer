# Deep Literary Style Transfer

This project rewrites a given text into the distinct stylistic voice of a famous author. It goes beyond simple word replacement by performing a deep, data-driven analysis of the author's style—including sentence structure, grammatical patterns, pacing, and punctuation—and then uses this analysis to guide a large language model (LLM) in the rewriting process.



## Features

-   **Deep Stylistic Analysis**: Uses `spaCy` to extract quantitative metrics for:
    -   Sentence length (average and variance)
    -   Vocabulary richness (Type-Token Ratio)
    -   Part-of-Speech (POS) distribution
    -   Punctuation frequency
    -   Readability scores (Flesch Reading Ease)
-   **LLM-Powered Transfer**: Feeds the detailed analysis to an LLM (via `Ollama`) as a strict set of instructions, ensuring a more authentic transfer of style.
-   **Content Preservation**: The LLM is prompted to retain the original meaning and intent of the source text, modifying only the style.
-   **Command-Line Interface**: Simple, easy-to-use CLI for analyzing and rewriting texts.

## How It Works

1.  **Analysis**: The `StyleAnalyzer` class takes a sample text from the target author (e.g., a chapter from a book). It uses `spaCy` to parse the text and calculate a vector of stylistic features.
2.  **Prompt Engineering**: The calculated features are formatted into a detailed, human-readable "style guide". This guide is the core of the project. Instead of a vague prompt like "Write this like Hemingway," the system generates a specific, constraint-based prompt (e.g., "*Average sentence length must be 15.2 words, ADJ/VERB ratio should be 0.4, and the text must have a Flesch Reading Ease score of ~85*").
3.  **Generation**: The `StyleTransfer` class sends the source text and the generated style guide to a local LLM running via Ollama. The LLM's task is to rewrite the source text while meticulously adhering to the stylistic constraints.

This method forces the LLM to focus on the deep grammatical and rhythmic signatures of the author, resulting in a more nuanced and authentic style transfer.

## Installation

1.  **Prerequisites**:
    -   Python 3.8+
    -   [Ollama](https://ollama.com/) installed and running. Make sure you have pulled a model, for example:
        bash
        ollama pull llama3
        

2.  **Clone the repository**:
    bash
    git clone https://github.com/bagait/deep-literary-style-transfer.git
    cd deep-literary-style-transfer
    

3.  **Install Python dependencies**:
    bash
    pip install -r requirements.txt
    

4.  **Download the spaCy model**:
    bash
    python -m spacy download en_core_web_sm
    

## Usage

Create two text files:
-   A `source.txt` file containing the text you want to rewrite.
-   A `style_ref.txt` file containing a sample of the target author's writing (a few paragraphs or a page is usually enough).

Then, run the script from your terminal.

### Example

Let's rewrite a verbose paragraph from H.P. Lovecraft into the terse, direct style of Ernest Hemingway.

1.  Create `source_lovecraft.txt`:
    text
    It is an unfortunate fact that the bulk of humanity is too limited in its mental vision to weigh with patience and intelligence those isolated phenomena, seen and felt only by a psychologically sensitive few, which lie outside its common experience. Men of broader intellect know that there is no sharp distinction betwiwxt the real and the unreal; that all things appear as they do only by virtue of the delicate individual physical and mental media through which we are made conscious of them; but the prosaic materialism of the majority condemns as madness the flashes of supersight which penetrate the common veil of obvious empiricism.
    

2.  Create `style_hemingway.txt`:
    text
    The hills across the valley of the Ebro were long and white. On this side there was no shade and no trees and the station was between two lines of rails in the sun. Close against the side of the station there was the warm shadow of the building and a curtain, made of strings of bamboo beads, hung across the open door into the bar, to keep out flies. The American and the girl with him sat at a table in the shade, outside the building. It was very hot and the express from Barcelona would come in forty minutes. It stopped at this junction for two minutes and went on to Madrid.
    

3.  Run the command:
    bash
    python main.py \
      --source source_lovecraft.txt \
      --style-ref style_hemingway.txt \
      --author "Ernest Hemingway" \
      --output hemingway_version.txt
    

4.  **Check the output** in `hemingway_version.txt`. The script will also print the generated style guide and the final result to the console.

### CLI Arguments

-   `--source`: (Required) Path to the source text file.
-   `--style-ref`: (Required) Path to the author's sample text for style analysis.
-   `--author`: (Required) Name of the target author.
-   `--output`: (Optional) Path to save the rewritten text. Defaults to `output.txt`.
-   `--model`: (Optional) Ollama model to use. Defaults to `llama3`.
-   `--ollama-host`: (Optional) Custom Ollama host URL (e.g., `http://192.168.1.10:11434`).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
