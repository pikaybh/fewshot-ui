import streamlit as st
from typing import (List, Tuple)
import logging


# Root 
logger_name = "sidebar"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)
# File Handler
file_handler = logging.FileHandler(f'logs/{logger_name}.log', encoding='utf-8-sig')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(r'%(asctime)s [%(name)s, line %(lineno)d] %(levelname)s: %(message)s'))
logger.addHandler(file_handler)
# Stream Handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(r'%(message)s'))
logger.addHandler(stream_handler)


# 로그인 상태 관리
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = ''

class MySideBar:
    def __init__(self) -> None:
        ...

    @classmethod
    def select_model(cls) -> List[str]:
        return ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]

    @staticmethod
    def profile() -> None:
        
        st.header("Profile Section")
        if not st.session_state.get("logged_in"):
            user_id = st.text_input("ID", value="snucem")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if user_id == st.secrets["admin"]["user"] and password == st.secrets["admin"]["password"]:  # 여기에 실제 인증 로직을 추가하세요
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]  # API Key를 세션 상태에 저장
                    logger.info(f"Logged in succefully!\nuser id: {user_id}")
                else:
                    st.error("Invalid ID or Password")
            # 로그인하지 않았을 때 API Key 입력 칸 추가
            st.session_state.api_key = st.text_input("API Key", type="password")  
            if st.button("Submit API Key") and not st.session_state.get("logged_in"):
                st.write(f"Logged in with API Key: {st.session_state.api_key}")
        else:
            st.markdown(f"**환영합니다!**\n- ID | {st.session_state.user_id}\n- Rank | admin")

    @staticmethod
    def zeroshot_model() -> Tuple[str]:
        st.header("Model Section")
        model = st.selectbox("Select Model", MySideBar.select_model())
        custom_prompt = st.text_area("Custom Prompt", value="당신은 사고 사례를 특정 작업공사로 분류하는 고급 AI입니다. 사고 사례의 설명을 받게 되면, 이를 사전에 정의된 작업공사에 따라 분류하는 것이 당신의 임무입니다. 용어와 문맥을 이해하여 최상의 분류를 수행하세요. 대답은 각설하고 해당하는 작업공사만 답합니다. 다음은 분류해야 할 작업공사 목록입니다:", height=250)
        return model, custom_prompt

    @staticmethod
    def fewshot_model() -> Tuple[str]:
        st.header("Model Section")
        model = st.selectbox("Select Model", MySideBar.select_model())
        custom_prompt = st.text_area("Custom Prompt", value="Fill in the missing value for the following example based on the few-shot learning examples:")
        num_shots = st.number_input("Number of Shots", min_value=1, max_value=100, value=5)
        return model, custom_prompt, num_shots

# Example usage of the MySideBar class
def main() -> None:
    MySideBar.profile()
    model, custom_prompt = MySideBar.zeroshot_model()
    model, custom_prompt, num_shots = MySideBar.fewshot_model()

# Main
if __name__ == "__main__":
    main()
