import streamlit as st
# from streamlit_autorefresh import st_autorefresh
import pandas as pd
import openai
from random import sample
import numpy as np
from io import BytesIO
import logging

# Root 
logger_name = "app"
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

# Streamlit UI
st.set_page_config(
    page_title='Automatic rifle | SNUCEM',
    page_icon='https://i.namu.wiki/i/NgVoid2KU7eIGUnYVeZKBcfdydT9zq9_l69cYGpP1LwOFKn4nnbHe_OhsE3MWPcDtt6jqST_9tUSjyuNw3lNzw.svg',
    initial_sidebar_state='collapsed'
)
st.title("Automatic rifle")
st.write("If shots are examples, this app is automatic rifle!")
# 로그인 상태 관리
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = ''

foo = '''
# 페이지 새로고침을 사용하여 입력 포커스를 옮기기 위한 JavaScript 삽입
st_autorefresh(interval=1000, key="focus_refresh")

def add_js_code():
    js_code = """
    <script>
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Tab') {
            event.preventDefault();
            var nextElement = document.querySelector('input[name="Password"]');
            if (nextElement) {
                nextElement.focus();
            }
        } else if (event.key === 'Enter' && document.activeElement.name === 'Password') {
            document.querySelector('button[aria-label="Login"]').click();
        }
    });
    </script>
    """
    st.components.v1.html(js_code)

add_js_code()
'''
# 사이드바에 프로필 섹션 추가
with st.sidebar:
    st.header("Profile Section")
    if not st.session_state.get("logged_in"):
        user_id = st.text_input("ID")
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
    
    # 사이드바에 모델 섹션 추가
    st.header("Model Section")
    model = st.selectbox("Select Model", ["gpt-3", "gpt-3.5-turbo", "gpt-4", "gpt-4o"])
    num_shots = st.number_input("Number of Shots", min_value=1, max_value=100, value=3)
    custom_prompt = st.text_area("Custom Prompt")

# 엑셀 파일 업로드
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

def replace_semicolon_with_comma(df):
    # 모든 셀에 대해 문자열 변환을 시도합니다.
    return df.map(lambda x: str(x).replace(';', ',') if pd.notnull(x) else x)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df = replace_semicolon_with_comma(df)
    st.write("DataFrame Loaded:")
    st.dataframe(df)
    
    # 입력 열 선택
    input_column = st.selectbox("Select Input Column", df.columns)
    st.write("First 5 values of the input column:")
    st.write(df[input_column].head())
    
    # 출력 열 선택
    output_column = st.selectbox("Select Output Column", df.columns)
    
    # 값이 있는 셀과 공란 셀의 개수 보여주기
    filled_count = df[output_column].notna().sum()
    empty_count = df[output_column].isna().sum()
    
    st.write(f"Filled Cells: {filled_count}")
    st.write(f"Empty Cells: {empty_count}")

    # 값이 있는 셀의 내용 보여주기
    st.write("Filled Cells Content:")
    st.write(df[df[output_column].notna()][[input_column, output_column]].head())
    
    if st.button("Fill Missing Values"):
        # Progress bar 설정
        progress_bar = st.progress(0)
        
        # 공란을 채우기
        def fill_missing_values():
            for idx in df[df[output_column].isna()].index:
                examples = df[df[output_column].notna()].sample(num_shots)
                prompt = custom_prompt if custom_prompt else "\n".join([f"{input_column}: {row[input_column]}, {output_column}: {row[output_column]}" for _, row in examples.iterrows()])
                messages = [
                    {
                        'role': 'system',
                        'content': "Fill in the missing value for the following example based on the few-shot learning examples:"
                    },
                    {
                        'role': 'user',
                        'content': f"{prompt}\n{input_column}: {df.at[idx, input_column]}, {output_column}:"
                    }
                ]

                response = openai.chat.completions.create(
                    model=model,
                    messages=messages
                )
                answer = response.choices[0].message.content.strip()
                df.at[idx, output_column] = answer.split(f"{output_column}: ")[-1] if f"{output_column}: " in answer else answer
                
                # 진행 상황 업데이트
                progress = (df[output_column].notna().sum() / (filled_count + empty_count))
                progress_bar.progress(progress)
                st.write(f"Progress: {df[output_column].notna().sum()} / {filled_count + empty_count}")
        
        fill_missing_values()
        
        st.write("All missing values have been filled.")
        st.dataframe(df)
        
        # 엑셀 파일 다운로드
        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
            processed_data = output.getvalue()
            return processed_data
        
        processed_data = to_excel(df)
        
        st.download_button(
            label="Download Excel file",
            data=processed_data,
            file_name="filled_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
