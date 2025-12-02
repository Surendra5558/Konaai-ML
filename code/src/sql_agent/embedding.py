# # Copyright (C) KonaAI - All Rights Reserved
"""TextEmbedder Module"""
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from src.utils.conf import Setup
from src.utils.status import Status


class TextEmbedder:
    """TextEmbedder is a utility class for generating text embeddings using a SentenceTransformer model.
    This class handles model initialization, downloading the model if necessary, and provides a method to generate embedding vectors for input text.
    Attributes:
        model (Optional[SentenceTransformer]): The underlying SentenceTransformer model used for embedding.
    """

    model: Optional[SentenceTransformer] = None

    def __init__(self):
        model_config_file = Path(self._model_path, "config.json")
        # Check if the model configuration file exists
        if not model_config_file.exists():
            self._download_model()

        if not model_config_file.exists():
            raise FileNotFoundError(
                f"Model configuration file not found at {model_config_file.as_posix()} after download."
            )

        Status.INFO(f"Loading embedding model from {self._model_path.as_posix()}.")

        self.model = SentenceTransformer(model_name_or_path=self._model_path.as_posix())

    def embed(self, text: Union[str, List[str]]) -> Optional[np.ndarray]:
        """
        Generates an embedding vector for the given text using the underlying model.

        Args:
            text (str): The input text to be embedded.

        Returns:
            np.ndarray: The embedding vector representation of the input text.
        """
        try:
            return self.model.encode(sentences=text, convert_to_numpy=True)
        except Exception as e:
            Status.FAILED(f"Error occurred while embedding text: {e}", traceback=True)
            return None

    @property
    def _model_path(self) -> Path:
        """
        Returns the path to the model file.
        If the model is not downloaded, it downloads the model and returns the path.
        """
        path = Path(Setup.user_data_dir(), "embedder").absolute()
        # create the directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _download_model(self) -> Path:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        path = snapshot_download(
            repo_id=model_name,
            revision="main",
            local_dir=self._model_path.as_posix(),
            allow_patterns=[
                "*.bin",
                "*.json",
                "*.txt",
            ],  # Only download weights and config files
        )
        return Path(path)


if __name__ == "__main__":
    embedder = TextEmbedder()
    result = embedder.embed("This is a test sentence for embedding.")
    print(type(result))
