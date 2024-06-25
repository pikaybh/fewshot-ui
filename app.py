# Internal Modules
from routes import (zero_shot, few_shot)
# External Modules
import streamlit as st
import logging

# Root 
logger_name = "app"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# Check if handler exists
if not logger.hasHandlers():
    # File Handler
    file_handler = logging.FileHandler(f'logs/{logger_name}.log', encoding='utf-8-sig')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(r'%(message)s'))
    logger.addHandler(file_handler)
    # Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(r'%(message)s'))
    logger.addHandler(stream_handler)


def main() -> None:
    # Streamlit UI
    st.set_page_config(
        page_title='LLM Helper | SNUCEM',
        page_icon='https://i.namu.wiki/i/NgVoid2KU7eIGUnYVeZKBcfdydT9zq9_l69cYGpP1LwOFKn4nnbHe_OhsE3MWPcDtt6jqST_9tUSjyuNw3lNzw.svg',
    )

    # 네비게이션 바
    navigation_container = st.container()
    page = navigation_container.radio("Navigation", ["Zero Shot", "Few Shot"], horizontal=True)

    if page == "Zero Shot":
        zero_shot.zero_shot_page()
    elif page == "Few Shot":
        few_shot.few_shot_page()

# Main
if __name__ == "__main__":
    main()
