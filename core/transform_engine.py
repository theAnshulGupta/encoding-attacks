from typing import Dict, List
import numpy as np

class TransformEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.word_map = {}
        self.context_patterns = {}
        
    @staticmethod
    def create(transform_type: str, model_name: str) -> 'TransformEngine':
        if transform_type == "word":
            return WordTransformEngine(model_name)
        elif transform_type == "contextual":
            return ContextTransformEngine(model_name)
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")
            
    def generate_teaching_example(self) -> Dict:
        text = self._generate_text()
        transformed = self._transform_text(text)
        loss = self._calculate_loss(text, transformed)
        return {
            "text": text,
            "transformed": transformed,
            "loss": loss
        }
        
    def generate_practice_example(self) -> Dict:
        text = self._generate_text()
        transformed = self._transform_text(text)
        loss = self._calculate_loss(text, transformed)
        return {
            "text": text,
            "transformed": transformed,
            "loss": loss
        }
        
    def transform_query(self, query: str) -> Dict:
        transformed = self._transform_text(query)
        loss = self._calculate_loss(query, transformed)
        return {
            "query": query,
            "transformed": transformed,
            "loss": loss
        }
        
    def update_from_teaching(self, result: Dict, learning_rate: float):
        self._update_mappings(result["text"], result["transformed"], learning_rate)
        
    def update_from_practice(self, result: Dict, learning_rate: float):
        self._update_mappings(result["text"], result["transformed"], learning_rate)
        
    def update_from_target(self, result: Dict, learning_rate: float):
        self._update_mappings(result["query"], result["transformed"], learning_rate)
        
    def _generate_text(self) -> str:
        return "This is a sample text"
        
    def _transform_text(self, text: str) -> str:
        words = text.split()
        transformed_words = []
        for word in words:
            transformed = self._transform_word(word)
            transformed_words.append(transformed)
        return " ".join(transformed_words)
        
    def _transform_word(self, word: str) -> str:
        if word in self.word_map:
            return self.word_map[word]
        transformed = word[::-1]  # Simple reversal as default
        self.word_map[word] = transformed
        return transformed
        
    def _calculate_loss(self, original: str, transformed: str) -> float:
        return np.random.random()  # Placeholder for actual loss calculation
        
    def _update_mappings(self, original: str, transformed: str, learning_rate: float):
        original_words = original.split()
        transformed_words = transformed.split()
        for orig, trans in zip(original_words, transformed_words):
            if orig in self.word_map:
                current = self.word_map[orig]
                self.word_map[orig] = trans if np.random.random() < learning_rate else current
            else:
                self.word_map[orig] = trans

class WordTransformEngine(TransformEngine):
    def _transform_word(self, word: str) -> str:
        if word in self.word_map:
            return self.word_map[word]
        if len(word) <= 1:
            return word
        transformed = word[1:] + word[0]  # Rotate first character to end
        self.word_map[word] = transformed
        return transformed

class ContextTransformEngine(TransformEngine):
    def __init__(self, model_name: str, context_window: int = 3):
        super().__init__(model_name)
        self.context_window = context_window
        
    def _transform_text(self, text: str) -> str:
        words = text.split()
        transformed_words = []
        for i in range(len(words)):
            context = self._get_context(words, i)
            word = words[i]
            key = (word, context)
            if key in self.context_patterns:
                transformed_words.append(self.context_patterns[key])
            else:
                transformed = self._transform_with_context(word, context)
                self.context_patterns[key] = transformed
                transformed_words.append(transformed)
        return " ".join(transformed_words)
        
    def _get_context(self, words: List[str], position: int) -> str:
        start = max(0, position - self.context_window)
        end = min(len(words), position + self.context_window + 1)
        context_words = words[start:position] + words[position+1:end]
        return " ".join(context_words)
        
    def _transform_with_context(self, word: str, context: str) -> str:
        if not context:
            return self._transform_word(word)
        return word + "_" + context.split()[0]
