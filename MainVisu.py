import sqlite3

import numpy as np
import streamlit as st

from src.utils import Board, Grades
from src.utils.Angles import ANGLES
from src.utils.Grades import GRADES
from src.utils.Utils import load_name_and_description

st.title('Kilter AI')

connection = sqlite3.connect('./resources/databases/kilter.sqlite')

dict = load_name_and_description(connection)

col1, col2 = st.columns(2)

with col1:
    name = st.selectbox("Select board name", dict.keys())
    st.text(f"Selected board name: {name}")
    description = st.selectbox("Select board description", dict[name])
    st.text(f"Selected board description: {description}")
    if name is not None and description is not None :
        with col2:
            st.text("Board layout:")
            board = Board.create_board(connection, 'kilter', name, description)
            matrix = np.zeros(0)
            st.image(board.visualize_climb(matrix))

if st.button("Choose this board ") :
    angle = None
    grade = None

angle = st.selectbox("Select an angle : ", ANGLES)
st.text(f"Selected angle: {angle}")
grade = st.selectbox("Select a grade : ", GRADES.keys())
st.text(f"Selected grade: {grade}")



