import sqlite3
import unittest

import pandas as pd
import torch
from torch.onnx.symbolic_opset9 import tensor
from src.ai.generator.MultiHeadGenerator import compute_distance_stats
from src.ai.generator.MultiHeadGenerator import MultiHeadGenerator, train_model, save_model, load_model, \
    generate_sequences
from src.database.KilterTorchDataset import KilterTorchDataset
from src.utils.Board import create_board
from src.visualizer.Visualizer import Visualizer
from src.utils.Climb import Climb

class MultiHeadGeneratorTest(unittest.TestCase):

    def test_train_model(self):
        dataset_path = "resources/databases/boards/16 x 12_Super Wide.csv"
        # Load dataset
        dataset = KilterTorchDataset(dataset_path)
        stats = compute_distance_stats(dataset)

        # Train model
        model = train_model(dataset, stats, MultiHeadGenerator, num_epochs=10, device="cpu")

        save_model(model, "resources/models/boards/16 x 12_Super Wide.pth")

    def test_generate_sequences(self):
        dataset_path = "resources/databases/boards/16 x 12_Super Wide.csv"
        # Load dataset
        dataset = KilterTorchDataset(dataset_path)
        stats = compute_distance_stats(dataset)

        # Later, when generating:
        model = load_model(
            MultiHeadGenerator,
            vocab_size=dataset.vocab_size,
            path="resources/models/boards/16 x 12_Super Wide.pth",
            device="cpu"
        )

        # Get one sample
        angle = 45.
        difficulty = 18. #Should be 6b-ish
        sample = [angle, difficulty]
        # Generate predictions
        preds = generate_sequences(model, torch.tensor(sample), stats, device="cpu")

        print(preds)

        db_path = "resources/databases/kilter.sqlite"
        connection = sqlite3.connect(db_path)
        name = "16 x 12"
        description = "Super Wide"
        board = create_board(connection, name, description)

        visualizer = Visualizer(board, connection)
        climb = Climb(preds['start'], preds['middle'], preds['finish'], preds['foot'], None, None, None, None)
        visualizer.visualize_climb(climb).show()















