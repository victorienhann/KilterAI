from src.utils import Utils

board_type = "kilter"
connection = Utils.connect_to_database(board_type)

name = "16 x 12"
description = "Super Wide"

dataset = Utils.load_dataset(name, description)

matrices = dataset["matrices"]

total_holds = 0
total_starts = 0
total_middles = 0
total_finishes = 0
total_feet = 0
for matrix in matrices:
    for line in matrix:
        for element in line:
            total_holds += 1
            if element == 1:
                total_starts += 1
            if element == 2:
                total_middles += 1
            if element == 3:
                total_finishes += 1
            if element == 4:
                total_feet += 1

print(f"Start hold proportion: {total_starts/total_holds:.10f}")
print(f"Middle hold proportion: {total_middles/total_holds:.10f}")
print(f"Finish hold proportion: {total_finishes / total_holds:.10f}")
print(f"Feet proportion: {total_feet/total_holds:.10f}")
print(f"Unused holds proportion: { 1 - (total_starts + total_middles + total_finishes + total_feet) / total_holds:.10f}")
