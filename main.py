import argparse
import spacy
import ollama
import numpy as np
from collections import Counter
import re
import sys

# Helper for syllable counting
def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

class StyleAnalyzer:
    """Analyzes the stylistic features of a text using spaCy."""

    def __init__(self, model="en_core_web_sm"):
        """Initializes the analyzer and loads the spaCy model."""
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Spacy model '{model}' not found.")
            print(f"Please run: python -m spacy download {model}")
            sys.exit(1)

    def analyze(self, text: str) -> dict:
        """Performs a deep stylistic analysis of the given text."""
        doc = self.nlp(text)

        if not doc or len(doc.text.strip()) == 0:
            raise ValueError("Input text is empty or contains only whitespace.")

        sentences = list(doc.sents)
        words = [token.text for token in doc if token.is_alpha]

        if not sentences or not words:
             raise ValueError("Could not extract sentences or words from the text.")

        # Basic stats
        total_words = len(words)
        total_sentences = len(sentences)
        total_syllables = sum(syllable_count(word) for word in words)

        # 1. Sentence Structure
        sentence_lengths = [len([token for token in sent if token.is_alpha]) for sent in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        std_sentence_length = np.std(sentence_lengths) if sentence_lengths else 0

        # 2. Vocabulary richness (Type-Token Ratio)
        ttr = len(set(words)) / total_words if total_words > 0 else 0

        # 3. Part-of-Speech (POS) distribution
        pos_counts = Counter(token.pos_ for token in doc if not token.is_punct and not token.is_space)
        pos_distribution = {pos: count / total_words for pos, count in pos_counts.items()} if total_words > 0 else {}

        # 4. Punctuation frequency
        punct_counts = Counter(token.text for token in doc if token.is_punct)
        total_puncts = sum(punct_counts.values())
        punct_distribution = {p: c / total_puncts for p, c in punct_counts.items()} if total_puncts > 0 else {}

        # 5. Readability (Flesch Reading Ease)
        # Formula: 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        flesch_ease = 0
        if total_sentences > 0 and total_words > 0:
            flesch_ease = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

        return {
            'avg_sentence_length': avg_sentence_length,
            'sentence_length_variance': std_sentence_length ** 2,
            'type_token_ratio': ttr,
            'pos_distribution': pos_distribution,
            'punctuation_distribution': punct_distribution,
            'flesch_reading_ease': flesch_ease,
        }

    def format_analysis_for_prompt(self, analysis: dict, author_name: str) -> str:
        """Formats the analysis dictionary into a human-readable prompt segment."""
        prompt = f"## Stylistic Analysis of {author_name}\n\n"
        prompt += f"- **Sentence Structure**: Emulate sentences with an average length of {analysis['avg_sentence_length']:.1f} words. The sentence length should be relatively consistent, with a variance of {analysis['sentence_length_variance']:.1f}.\n"
        prompt += f"- **Vocabulary**: Use a vocabulary of moderate richness (Type-Token Ratio of around {analysis['type_token_ratio']:.2f}). Avoid overly obscure or complex words.\n"
        prompt += f"- **Grammar (Part-of-Speech)**: The writing heavily features the following parts of speech. Replicate this distribution:\n"
        
        top_pos = sorted(analysis['pos_distribution'].items(), key=lambda item: item[1], reverse=True)[:5]
        for pos, percent in top_pos:
            prompt += f"  - {pos}: ~{percent:.1%}\n"

        prompt += f"- **Punctuation**: Punctuation usage is a key part of the style. Prioritize the following marks:\n"
        top_puncts = sorted(analysis['punctuation_distribution'].items(), key=lambda item: item[1], reverse=True)[:3]
        for p, percent in top_puncts:
             prompt += f"  - The '{p}' character comprises ~{percent:.1%} of all punctuation.\n"

        prompt += f"- **Clarity**: The overall style is clear and direct, with a Flesch Reading Ease score around {analysis['flesch_reading_ease']:.0f}. Aim for this level of readability.\n"
        return prompt


class StyleTransfer:
    """Uses an LLM to perform style transfer based on a stylistic analysis."""
    def __init__(self, model="llama3", host=None):
        """Initializes the Ollama client."""
        self.model = model
        self.client = ollama.Client(host=host)
        try:
            self.client.list()
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Please ensure Ollama is running and the model is installed (e.g., 'ollama run llama3').")
            sys.exit(1)

    def rewrite(self, source_text: str, style_guide: str, author_name: str) -> str:
        """Generates the rewritten text by prompting the LLM."""
        system_prompt = (
            "You are an expert literary editor. Your task is to rewrite a given text in the distinct style of a famous author. "
            "You will be provided with a detailed, data-driven stylistic analysis of the target author's work. "
            "You MUST adhere strictly to this analysis, focusing on sentence structure, grammar, and rhythm, not just word choice. "
            "Do NOT add new ideas, content, or plot points. Preserve the original meaning and intent of the source text perfectly. "
            "Rewrite ONLY the text provided."
        )

        user_prompt = (
            f"{style_guide}\n\n"
            f"## Task\n\n"
            f"Rewrite the following source text into the style of {author_name}. "
            f"Apply the stylistic rules from the analysis above meticulously.\n\n"
            f"### Source Text:\n\n`{source_text}`\n\n"
            f"### Rewritten Text in the style of {author_name}:"
        )
        
        print("\n--- Sending request to LLM... ---")
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        print("--- Received response. ---")
        return response['message']['content']


def main():
    parser = argparse.ArgumentParser(description="Deep Literary Style Transfer")
    parser.add_argument("--source", required=True, help="Path to the source text file.")
    parser.add_argument("--style-ref", required=True, help="Path to the author's sample text for style analysis.")
    parser.add_argument("--author", required=True, help="Name of the target author (e.g., 'Ernest Hemingway').")
    parser.add_argument("--output", default="output.txt", help="Path to save the rewritten text.")
    parser.add_argument("--model", default="llama3", help="Ollama model to use for generation.")
    parser.add_argument("--ollama-host", default=None, help="Optional host for Ollama (e.g., http://localhost:11434)")
    args = parser.parse_args()

    try:
        with open(args.source, 'r', encoding='utf-8') as f:
            source_text = f.read()
        with open(args.style_ref, 'r', encoding='utf-8') as f:
            style_ref_text = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 1. Analyze the style of the reference author
    print(f"Analyzing the style of {args.author} from '{args.style_ref}'...")
    analyzer = StyleAnalyzer()
    try:
        style_analysis = analyzer.analyze(style_ref_text)
    except ValueError as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)
    
    # 2. Format the analysis into a detailed prompt guide
    style_guide = analyzer.format_analysis_for_prompt(style_analysis, args.author)
    print("\n--- Generated Style Guide ---")
    print(style_guide)
    print("---------------------------")

    # 3. Perform the style transfer
    transfer = StyleTransfer(model=args.model, host=args.ollama_host)
    rewritten_text = transfer.rewrite(source_text, style_guide, args.author)

    # 4. Display and save the result
    print(f"\n--- Original Text ---")
    print(source_text)
    print(f"\n--- Rewritten in the style of {args.author} ---")
    print(rewritten_text)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(rewritten_text)
    print(f"\nResult saved to '{args.output}'")

if __name__ == "__main__":
    main()
