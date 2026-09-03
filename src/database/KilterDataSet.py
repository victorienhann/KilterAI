class KilterDataset:
    """A collection of climbs encoded as token vectors over a board's hold vocabulary
    (see Board.create_token_vector), each paired with its angle and grade."""

    def __init__(self, tokens, angles, grades):
        self.tokens = tokens
        self.angles = angles
        self.grades = grades

    def get_tokens(self):
        return self.tokens

    def get_angles(self):
        return self.angles

    def get_grades(self):
        return self.grades
