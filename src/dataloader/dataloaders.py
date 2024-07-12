import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import spacy
import unittest
import random

# Load SpaCy model (you may need to download it first with: python -m spacy download en_core_web_trf)
nlp = spacy.load("en_core_web_trf")

class LLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class BaseDataset:
    def __init__(self):
        pass


class WikiGermanDataset(BaseDataset):
    def __init__(self):
        pass

    def build_vocabulary(self, dataset):
        vocab = set()
        for example in dataset:
            doc = nlp(example["text"])
            vocab.update([token.text.lower() for token in doc])
        vocab.add("<PAD>")
        vocab.add("<UNK>")
        vocab.add("<DOC_START>")
        vocab.add("<DOC_END>")
        return vocab

    def preprocess_and_concatenate(self, dataset, token_to_idx, max_len=512):
        concatenated_inputs = []
        current_sequence = []
        
        for example in dataset:
            doc = nlp(example["text"])
            tokens = [token_to_idx["<DOC_START>"]] + [token_to_idx.get(token.text.lower(), token_to_idx["<UNK>"]) for token in doc] + [token_to_idx["<DOC_END>"]]
            
            while tokens:
                space_left = max_len - len(current_sequence)
                current_sequence.extend(tokens[:space_left])
                tokens = tokens[space_left:]
                
                if len(current_sequence) == max_len:
                    concatenated_inputs.append(torch.tensor(current_sequence))
                    current_sequence = []

        # Add the last sequence if it's not empty
        if current_sequence:
            concatenated_inputs.append(torch.tensor(current_sequence + [token_to_idx["<PAD>"]] * (max_len - len(current_sequence))))

        return concatenated_inputs

    def tokenize(self, text):
        """
        Tokenize the input text using SpaCy.
        
        Args:
        text (str): Input text to tokenize.
        
        Returns:
        list: List of tokens.
        """
        doc = nlp(text)
        return [token.text.lower() for token in doc]

    def detokenize(self, tokens):
        """
        Convert tokens back to text.
        
        Args:
        tokens (list): List of tokens.
        
        Returns:
        str: Reconstructed text.
        """
        return ' '.join(tokens)


class TestingDatasetWikiGerman(WikiGermanDataset):
    def __init__(self, test_file_path, batch_size=1, max_len=512):
        self.batch_size = batch_size
        self.max_len = max_len
        
        # Load test sentences
        with open(test_file_path, 'r') as file:
            self.test_sentences = [line.strip() for line in file if line.strip()]
        
        # Create a dataset-like structure for test sentences
        self.dataset = [{"text": sentence} for sentence in self.test_sentences]
        
        # Process test sentences
        self.vocab = self.build_vocabulary(self.dataset)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.processed_data = self.preprocess_and_concatenate(self.dataset, self.token_to_idx, self.max_len)
        self.custom_dataset = LLDataset(self.processed_data)
        self.train_loader = DataLoader(self.custom_dataset, batch_size=self.batch_size, shuffle=True)

    def __iter__(self):
        return iter(self.train_loader)

    def __len__(self):
        return len(self.train_loader)

    def get_vocab_size(self):
        return len(self.vocab)

    def get_batch_size(self):
        return self.batch_size

    def get_max_len(self):
        return self.max_len

    def get_original_sentences(self):
        return self.test_sentences

class WikiGermanDataLoaderTrain(WikiGermanDataset):
    def __init__(self, dataset_name="gwlms/dewiki-20230701-flair-corpus", subset_size=1024, batch_size=16, max_len=512):
        self.dataset = load_dataset(dataset_name)
        if subset_size:
            self.dataset = self.dataset["train"].select(range(subset_size))
        else:
            self.dataset = self.dataset["train"]

        self.vocab = self.build_vocabulary(self.dataset)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        self.processed_data = self.preprocess_and_concatenate(self.dataset, self.token_to_idx, max_len)
        self.custom_dataset = LLDataset(self.processed_data)
        self.train_loader = DataLoader(self.custom_dataset, batch_size=batch_size, shuffle=True)
        
class TestLLMDataloader(unittest.TestCase):
    def setUp(self):
        self.llm_dataloader = WikiGermanDataLoaderTrain(subset_size=100, batch_size=25, max_len=512)
        with open('./data/raw_test/test.txt', 'r') as file:
            self.test_sentences = [line.strip() for line in file if line.strip()]


    def test_dataloader(self):
        print(f"Length of processed_data: {len(self.llm_dataloader.processed_data)}")
        print(f"Vocabulary size: {len(self.llm_dataloader.vocab)}")

        for batch in self.llm_dataloader.train_loader:
            print("Batch shape:", batch.shape)
            break  # Print only the first batch

        self.assertTrue(len(self.llm_dataloader.processed_data) > 0)
        self.assertTrue(len(self.llm_dataloader.vocab) > 0)

    def test_tokenize_detokenize(self):
        for sentence in self.test_sentences:
            tokens = self.llm_dataloader.tokenize(sentence)
            reconstructed = self.llm_dataloader.detokenize(tokens)
            
            print(f"Original: {sentence}")
            print(f"Tokenized: {tokens}")
            print(f"Reconstructed: {reconstructed}")
            print()

            # Check if the reconstructed sentence is the same as the original
            # Note: This comparison is case-insensitive and ignores punctuation
            self.assertEqual(
                ''.join(char.lower() for char in sentence if char.isalnum()),
                ''.join(char.lower() for char in reconstructed if char.isalnum())
            )

    def test_vocabulary_building(self):
        all_tokens = set()
        for sentence in self.test_sentences:
            tokens = self.llm_dataloader.tokenize(sentence)
            all_tokens.update(tokens)
        
        print(f"Test Vocabulary size: {len(all_tokens)}")
        print(f"Test Vocabulary: {sorted(all_tokens)}")

        # Check if the vocabulary size is reasonable
        self.assertTrue(20 <= len(all_tokens) <= 50)  # Adjust these bounds as needed