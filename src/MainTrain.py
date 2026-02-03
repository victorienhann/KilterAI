
from torch.utils.data import DataLoader

from src.ai.generator.VariationalAutoEncoderBis import VariationalAutoEncoderBis, trainBis, TrainingDatasetBis, \
    generateBis
from src.ai.generator.VariationalAutoencoder import VariationalAutoEncoder, TrainingDataset, generate
from src.ai.generator.VariationalAutoencoder import train
from src.utils import Utils, Board
from src.utils.Grades import GRADES
from src.utils.Utils import save_model

name = "16 x 12"
description = "Super Wide"
dataset = TrainingDataset(name, description)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

datasetBis = TrainingDatasetBis(name, description)
loaderBis = DataLoader(datasetBis, batch_size=32, shuffle=True)

bis = False

H, W = dataset.H, dataset.W
angle_min, angle_max, grades_min, grades_max = dataset.angles_min, dataset.angles_max, dataset.grades_min, dataset.grades_max
angle = 35
grade = "6b"


bis = False
if not bis :
    model = VariationalAutoEncoder(H, W, angle_min, angle_max, grades_min, grades_max, latent_dim=16)
    train(model, loader, epochs=30)
    matrix = generate(model, angle, GRADES[grade])
else :
    model = VariationalAutoEncoderBis(H, W, angle_min, angle_max, grades_min, grades_max, latent_dim=16)
    trainBis(model, loaderBis, epochs=10)
    matrix = generateBis(model, angle, GRADES[grade])

save_model(model, name, description)

board_type = "kilter"
connection = Utils.connect_to_database(board_type)
board = Board.create_board(connection, board_type, name, description)
board.visualize_climb(matrix).show()
