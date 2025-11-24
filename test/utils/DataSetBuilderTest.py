import unittest

import pandas as pd

import src.utils.DataSetBuilder as dataset_builder

class DataSetBuilderTest(unittest.TestCase):

    def test_connect(self):
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        self.assertTrue(builder.connect())

        db_path_fail = "resources/databases/kilter_fail.sqlite" # does not exist, connection should fail
        builder_fail = dataset_builder.DataSetBuilder(db_path_fail)
        with self.assertRaises(FileNotFoundError):
            builder_fail.connect()

    def test_build_dataset(self):
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        dataset = builder.build_dataset(layout_id=1)

    def test_extract_roles(self):
        climb_valid = "p1234r12p2345r13p3456r14p4567r15p3456r12"
        climb_invalid = "p1234r67"
        start, middle, finish, foot = dataset_builder.extract_roles(climb_valid)

        self.assertTrue(start[0] == 1234 and len(start) == 2)
        self.assertTrue(middle[0] == 2345 and len(middle) == 1)
        self.assertTrue(finish[0] == 3456 and len(finish) == 1)
        self.assertTrue(foot[0] == 4567 and len(foot) == 1)
        with self.assertRaises(ValueError):
            dataset_builder.extract_roles(climb_invalid)

    def test_export_dataset(self):
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        dataset = builder.build_dataset(1)
        dataset.export_to_csv("resources/databases/kilter.csv")

    def test_save_climbs(self):
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        connection = builder.connect()
        query = """SELECT * FROM climbs WHERE name = 'SAD' """
        df = pd.read_sql_query(query, connection)
        df.to_csv("resources/databases/kilter.csv", index=False)

    def test_create_dataset_board(self):
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        description = "Super Wide"
        name = "16 x 12"
        builder.build_dataset(name, description)

    def test_export_datasets(self):
        db_path = "resources/databases/kilter.sqlite"
        builder = dataset_builder.DataSetBuilder(db_path)
        query = """ SELECT name, description FROM product_sizes"""
        df = pd.read_sql_query(query, builder.connect())
        names = [df["name"].values[i] for i in range(len(df["name"]))]
        descriptions = [df["description"].values[i] for i in range(len(df["description"]))]
        for i in range(len(names)):
            name = names[i]
            description = descriptions[i]
            dataset = builder.build_dataset(name, description)
            filename = "resources/databases/boards/" + name + "_" + description + ".csv"
            dataset.export_to_csv(filename)












