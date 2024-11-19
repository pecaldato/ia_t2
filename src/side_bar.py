import streamlit as st
import json


class SideBar:

    def __init__(self, options_path):

        with open(options_path, "r") as file:
            self.options = json.load(file)
        self.options = sorted([opt for opt in self.options.values()])

        self.mask_types = ["Black", "White", "Blur"]

        self.selected_class = None
        self.selected_mask = None
        self.selected_mask_region = None

        pass

    def generateSideBar(self):
        st.sidebar.header("Options:")
        self.selected_class = st.sidebar.selectbox("Select a class to segment:", self.options)

        self.selected_mask_region = st.sidebar.radio("Select the mask region:", ["Object", "Background"])

        self.selected_mask = st.sidebar.selectbox("Select the mask type:", self.mask_types)

        return self.selected_class, self.selected_mask, self.selected_mask_region