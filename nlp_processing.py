import spacy
from nltk.stem import PorterStemmer
from typing import List, Tuple, Dict, Any
from logger import setup_logger

class NLPProcessor:
    """A class to handle NLP preprocessing tasks using spaCy and NLTK."""
    
    def __init__(self):
        """Initialize spaCy model and NLTK stemmer."""
        self.logger = setup_logger(__name__)
        self.logger.info("Initializing NLPProcessor")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
            self.stemmer = PorterStemmer()
            self.logger.info("Successfully loaded spaCy model and NLTK stemmer")
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP tools: {str(e)}")
            raise RuntimeError(f"Failed to initialize NLP tools: {str(e)}")

    def _validate_input(self, text: str) -> None:
        """Validate input text."""
        self.logger.debug(f"Validating input: {text}")
        if not isinstance(text, str) or not text.strip():
            self.logger.warning("Invalid input: text is empty or not a string")
            raise ValueError("Input text must be a non-empty string")

    def _process_text(self, text: str) -> spacy.tokens.Doc:
        """Process text with spaCy and return a Doc object."""
        self.logger.info(f"Processing text with spaCy: {text}")
        self._validate_input(text)
        try:
            doc = self.nlp(text)
            self.logger.debug(f"spaCy Doc created with {len(doc)} tokens")
            return doc
        except Exception as e:
            self.logger.error(f"Error processing text with spaCy: {str(e)}")
            raise RuntimeError(f"Error processing text with spaCy: {str(e)}")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into individual words or tokens."""
        self.logger.info("Tokenizing text")
        doc = self._process_text(text)
        tokens = [token.text for token in doc]
        self.logger.debug(f"Tokens: {tokens}")
        return tokens

    def lemmatize(self, text: str) -> List[str]:
        """Lemmatize text to reduce words to their base dictionary form."""
        self.logger.info("Lemmatizing text")
        doc = self._process_text(text)
        lemmas = [token.lemma_ for token in doc]
        self.logger.debug(f"Lemmas: {lemmas}")
        return lemmas

    def stem(self, text: str) -> List[str]:
        """Stem text to reduce words to their root form using Porter Stemmer."""
        self.logger.info("Stemming text")
        tokens = self.tokenize(text)
        try:
            stems = [self.stemmer.stem(token) for token in tokens]
            self.logger.debug(f"Stems: {stems}")
            return stems
        except Exception as e:
            self.logger.error(f"Error stemming text: {str(e)}")
            raise RuntimeError(f"Error stemming text: {str(e)}")

    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """Perform Part-of-Speech (POS) tagging on text."""
        self.logger.info("Performing POS tagging")
        doc = self._process_text(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        self.logger.debug(f"POS tags: {pos_tags}")
        return pos_tags

    def ner(self, text: str) -> List[Tuple[str, str]]:
        """Perform Named Entity Recognition (NER) on text."""
        self.logger.info("Performing Named Entity Recognition")
        doc = self._process_text(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        self.logger.debug(f"Named entities: {entities}")
        return entities

    def compare_lemmas_stems_realtime(self, text: str) -> Dict[str, Any]:
        """Compare lemmatization and stemming for words in the input text and explain differences."""
        self.logger.info("Generating real-time lemma vs. stem comparison")
        doc = self._process_text(text)
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        stems = [self.stemmer.stem(token) for token in tokens]
        
        # Create comparison list, excluding punctuation and duplicates
        comparison = []
        seen_words = set()
        for token, lemma, stem in zip(tokens, lemmas, stems):
            if token.isalpha() and token.lower() not in seen_words:
                difference = (
                    "Same" if lemma.lower() == stem.lower() else
                    "Different: Lemma uses context and part-of-speech; stem uses rule-based suffix stripping"
                )
                comparison.append({
                    "word": token,
                    "lemma": lemma,
                    "stem": stem,
                    "difference": difference
                })
                seen_words.add(token.lower())
        
        self.logger.debug(f"Real-time comparison: {comparison}")
        
        # Generate explanation
        explanation = [
            "Lemmatization vs Stemming Comparison:",
            "",
            "- **Lemmatization**: Reduces words to their base dictionary form (lemma) using linguistic rules and context.",
            "- **Stemming**: Strips suffixes using heuristic rules, often producing non-words.",
            "",
            "Analysis of Your Input:"
        ]
        
        if not comparison:
            explanation.append("- No valid alphabetic words found in the input to compare.")
        else:
            for comp in comparison:
                explanation.append(f"- Word: '{comp['word']}':")
                explanation.append(f"  - Lemma: '{comp['lemma']}' ({comp['difference']})")
                explanation.append(f"  - Stem: '{comp['stem']}'")
        
        explanation.extend([
            "",
            "**Lemmatization Details**:",
            "- Considers part-of-speech and context, ensuring valid dictionary words (e.g., 'running' → 'run', 'better' → 'good').",
            "- Pros: Accurate, meaningful words, context-aware.",
            "- Cons: Slower, requires robust linguistic resources.",
            "- Use Cases: Semantic analysis, machine translation.",
            "",
            "**Stemming Details**:",
            "- Faster but less precise, may produce invalid roots (e.g., 'studies' → 'studi', 'geese' → 'gees').",
            "- Pros: Fast, simple, effective for reducing variations.",
            "- Cons: Less accurate, context-agnostic.",
            "- Use Cases: Search engines, text indexing.",
            "",
            "**Key Differences**:",
            "- Lemmatization ensures valid words; stemming may not.",
            "- Lemmatization is context-aware; stemming is rule-based.",
            "- Lemmatization is slower but precise; stemming is faster but cruder."
        ])
        
        explanation_text = "\n".join(explanation)
        self.logger.debug(f"Real-time explanation: {explanation_text}")
        return {
            "comparison": comparison,
            "explanation": explanation_text
        }

    def compare_lemmatization_stemming(self) -> Dict[str, Any]:
        """Compare lemmatization and stemming for a fixed set of words."""
        self.logger.info("Generating static lemmatization vs. stemming comparison")
        words = [
            "running", "ran", "runs",
            "studies", "studying",
            "geese", "children",
            "easily", "fairly",
            "better", "best",
            "organization"
        ]
        
        comparison = []
        for word in words:
            try:
                lemma = self.nlp(word)[0].lemma_
                stem = self.stemmer.stem(word)
                comparison.append({
                    "word": word,
                    "lemma": lemma,
                    "stem": stem
                })
            except Exception as e:
                self.logger.error(f"Error processing word '{word}': {str(e)}")
                comparison.append({
                    "word": word,
                    "lemma": f"Error: {str(e)}",
                    "stem": f"Error: {str(e)}"
                })

        self.logger.debug(f"Static comparison: {comparison}")
        
        explanation = """
        Lemmatization vs Stemming Comparison:

        **Lemmatization**:
        - Reduces words to their base or dictionary form (lemma) using linguistic rules and context.
        - Considers part-of-speech and context, ensuring valid dictionary words.
        - Examples: 'running' -> 'run', 'geese' -> 'goose', 'better' -> 'good'.
        - Pros: Highly accurate, produces meaningful words, context-aware.
        - Cons: Computationally intensive, requires robust linguistic resources.
        - Use Cases: Semantic analysis, machine translation, information retrieval.

        **Stemming**:
        - Strips suffixes using heuristic rules, often resulting in non-words.
        - Faster but less precise, may produce invalid or ambiguous roots.
        - Examples: 'running' -> 'run', 'geese' -> 'gees', 'studies' -> 'studi'.
        - Pros: Fast, simple, reduces word variations effectively.
        - Cons: Less accurate, may lose semantic meaning, context-agnostic.
        - Use Cases: Search engines, text indexing, basic text preprocessing.

        **Key Differences**:
        - Lemmatization ensures valid words; stemming may not (e.g., 'studies' -> 'studi').
        - Lemmatization uses context (e.g., 'better' -> 'good'); stemming is rule-based.
        - Lemmatization is slower but more precise; stemming is faster but cruder.
        - Lemmatization is better for tasks requiring semantic accuracy; stemming suits quick preprocessing.
        """
        
        self.logger.debug("Static explanation generated")
        return {"comparison": comparison, "explanation": explanation}