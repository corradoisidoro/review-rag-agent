import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.vector import initialize_retriever


@patch('app.vector.Chroma')
@patch('app.vector.OllamaEmbeddings')
@patch('app.vector.os.path.exists')
@patch('app.vector.pd.read_csv')
def test_initialize_retriever_adds_documents(mock_read_csv, mock_exists, mock_embed, mock_chroma):
    mock_exists.return_value = False

    mock_read_csv.return_value = pd.DataFrame({
        "Title": ["Good"], "Review": ["Tasty"], "Rating": [5], "Date": ["2023-01-01"]
    })

    mock_vs_instance = MagicMock()
    mock_chroma.return_value = mock_vs_instance

    # Execute
    retriever = initialize_retriever(
        "fake.csv", "fake_db", "mxbai-embed-large")

    mock_vs_instance.add_documents.assert_called_once()
    assert retriever is not None


@patch('app.vector.Chroma')
@patch('app.vector.OllamaEmbeddings')
@patch('app.vector.os.path.exists')
def test_initialize_retriever_skips_loading_if_exists(mock_exists, mock_embed, mock_chroma):
    mock_exists.return_value = True

    retriever = initialize_retriever(
        "fake.csv", "fake_db", "mxbai-embed-large")

    mock_chroma.assert_called_once()
    mock_chroma.return_value.add_documents.assert_not_called()
