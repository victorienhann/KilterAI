import numpy as np
import pandas as pd
from tqdm import tqdm


from src.database.KilterDataSet import KilterDataset
from src.utils.Colors import COLORS
from src.utils.Queries import QUERIES
from src.utils.Utils import extract_roles, make_circle, save_dataset
from src.visualizer import Visualizer


def create_board(connection, board_type, name, description):
    df = pd.read_sql(QUERIES["board_info"], connection, None, None,{'name': name, 'description': description})
    layout_id = int(df['layout_id'].values[0])
    set_id = [int(df['set_id'].values[i]) for i in range(len(df['set_id'].values))]
    edge_left = int(df['edge_left'].values[0])
    edge_right = int(df['edge_right'].values[0])
    edge_bottom = int(df['edge_bottom'].values[0])
    edge_top = int(df['edge_top'].values[0])
    return Board(connection, board_type, name, description, layout_id, set_id, edge_left, edge_right, edge_bottom, edge_top)

class Board:
    def __init__(self, connection, board_type, name, description, layout_id, set_id, edge_left, edge_right, edge_bottom, edge_top):
        self.connection = connection
        self.board_type = board_type
        self.name = name
        self.description = description
        self.layout_id = layout_id
        self.set_id = set_id
        self.edge_left = edge_left
        self.edge_right = edge_right
        self.edge_top = edge_top
        self.edge_bottom = edge_bottom
        # The board photo is only needed to actually draw a climb (visualize_climb);
        # dataset building and training never touch it, so it's fetched lazily by
        # load_template() rather than up front here - it requires network access to
        # the board vendor's API, which building/training shouldn't depend on.
        self.template = None
        self.mapping = None
        # Hold vocabulary: a fixed-size list of the board's actual hold positions.
        # Built lazily by load_holds() since both build_dataset() and visualize_climb()
        # need it, but only build_dataset() also needs the (much heavier) climbs query.
        self.hold_id_to_index = None
        self.hold_x = None
        self.hold_y = None
        self.vocab_size = None

    def get_edges(self):
        return self.edge_left, self.edge_right, self.edge_bottom, self.edge_top

    def load_template(self):
        """Fetch (and cache) the board's photo, downloading it if needed. Only
        visualize_climb() needs this - kept separate from __init__/create_board
        so building a Board for dataset building or training never requires it."""
        if self.template is not None:
            return
        self.template = Visualizer.get_template(self.board_type, self.name, self.description)

    def load_holds(self):
        """Fetch this board's hold positions and build the hold vocabulary (id -> slot index).

        The board has a fixed, small set of physical hold positions (~692 for the
        16x12 Super Wide kilter board); every climb only ever lights up a handful of
        them. We index climbs against this fixed vocabulary rather than a dense
        pixel grid, so this is cheap and safe to call whenever we just need the
        board layout (e.g. to visualize a climb) without paying for the full climbs
        query in extract_data().
        """
        if self.hold_x is not None:
            return
        self.mapping = pd.read_sql(QUERIES["holds"], self.connection, None, None,
                                   {'set_id_1' : self.set_id[0], 'set_id_2' : self.set_id[1], 'layout_id' : self.layout_id})
        id_to_index = {}
        xs, ys = [], []
        for idx, row in enumerate(self.mapping.itertuples(index=False)):
            placement_id, mirrored_id, x, y = row[0], row[1], row[2], row[3]
            # A climb's frames may reference either the regular placement id or its
            # mirrored counterpart depending on which set was used; both map to the
            # same physical hold, hence the same vocabulary slot.
            if pd.notna(placement_id):
                id_to_index[int(placement_id)] = idx
            if pd.notna(mirrored_id):
                id_to_index[int(mirrored_id)] = idx
            xs.append(x)
            ys.append(y)
        self.hold_id_to_index = id_to_index
        self.hold_x = np.array(xs)
        self.hold_y = np.array(ys)
        self.vocab_size = len(xs)

    def extract_data(self):
        """Extract climbs, angle and display_difficulty from the database"""
        try:
            df = pd.read_sql(QUERIES["board_climb"], self.connection, None, None, {"layout_id": self.layout_id,
                             "edge_left" : self.edge_left, "edge_right" : self.edge_right,
                             "edge_bottom" : self.edge_bottom, "edge_top" : self.edge_top})
            print(f"Found {len(df)} climbs for board {self.name} {self.description}")
            print("Mapping ...")
            self.load_holds()
            print(f"Mapping done ({self.vocab_size} holds)")
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def create_token_vector(self, start, middle, finish, foot):
        """Encode a climb as a vector over the fixed hold vocabulary: one entry per
        physical hold, valued 0 (unused) / 1 (start) / 2 (middle) / 3 (finish) / 4 (foot).
        """
        tokens = np.zeros(self.vocab_size, dtype=np.int8)
        for hold_ids, label in ((start, 1), (middle, 2), (finish, 3), (foot, 4)):
            for hold_id in hold_ids:
                idx = self.hold_id_to_index.get(hold_id)
                if idx is not None:
                    tokens[idx] = label
        return tokens

    def build_dataset(self):
        df = self.extract_data()
        climbs = df['frames']
        tokens = []
        angles = []
        grades = []

        for i, climb in enumerate(tqdm(climbs, desc=f"Building dataset for board {self.description} {self.name}" , unit="climbs")):
            starts_climb, middles_climb, finishes_climb, feet_climb = extract_roles(climb)
            if len(starts_climb) <= 2 and len(finishes_climb) <= 2:
                tokens.append(self.create_token_vector(starts_climb, middles_climb, finishes_climb, feet_climb))
                angles.append(df['angle'][i])
                grades.append(round(df['display_difficulty'][i]))

        dataset = KilterDataset(tokens, angles, grades)
        save_dataset(dataset, self.name, self.description)

    def visualize_climb(self, tokens):
        """Draw a climb given as a vector over the hold vocabulary (see create_token_vector)."""
        self.load_holds()
        self.load_template()

        colors = {
            1: COLORS["start"],
            2: COLORS["middle"],
            3: COLORS["finish"],
            4: COLORS["foot"],
        }

        edge_left, edge_right, edge_bottom, edge_top = self.get_edges()
        # Getting the board image without markers
        image = self.template
        width, height = self.template.size

        xSpacing = width / (edge_right - edge_left)
        ySpacing = height / (edge_top - edge_bottom)

        for idx, role in enumerate(tokens):
            if role > 0:
                x, y = self.hold_x[idx], self.hold_y[idx]
                xPixel = int((x - edge_left) * xSpacing)
                yPixel = int(height - (y - edge_bottom) * ySpacing)
                circle = make_circle(colors[int(role)])
                image.paste(circle, (xPixel - circle.width // 2, yPixel - circle.height // 2), circle)
        return image


