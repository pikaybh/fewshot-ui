# Internal Modules
from components.sidebar import MySideBar
from classes.message import Message
from utils.mypd import MyDataFrame as mdf
# External Modules
import streamlit as st
import pandas as pd
import openai
import numpy as np
from copy import deepcopy
from io import BytesIO
from typing import Union, List
import logging
# Root 
logger_name = "routes.zero-shot"
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

def zero_shot_page():
    st.write("### Zero Shot Page")
    
    # 사이드바에 프로필 섹션 추가
    with st.sidebar:
        MySideBar.profile()
        # 사이드바에 모델 섹션 추가
        model, custom_prompt = MySideBar.zeroshot_model()

    # 엑셀 파일 업로드
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df: Union[pd.DataFrame, mdf] = pd.read_excel(uploaded_file)
        df = mdf(df)
        df = df.replace_semicolon_with_comma()  # replace_semicolon_with_comma(df)
        st.write("DataFrame Loaded:")
        st.dataframe(df)
        
        # 입력 열 선택
        input_column = st.selectbox("Select Input Column", df.columns)
        st.write("First 5 values of the input column:")
        st.write(df[input_column].head())
        
        # 기존 열 선택 또는 새 열 이름 입력
        use_existing_column = st.checkbox("Use existing column", value=True)
        if use_existing_column:
            output_column = st.selectbox("Select Output Column", df.columns)
        else:
            output_column = st.text_input("Enter new column name")
            if output_column not in df.columns:
                df[output_column] = np.nan  # 새로운 열을 NaN 값으로 초기화

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

            # Messages
            messages: List[Message] = [
                Message(
                    **{
                        'role': 'system',
                        'content': custom_prompt
                    }
                )
            ]
            
            # 공란을 채우기
            def fill_missing_values():
                for idx in df[df[output_column].isna()].index:
                    tmp_messages: List[Message] = deepcopy(messages)
                    
                    tmp_messages.append(
                        Message(
                                **{
                                    'role': 'user',
                                    'content': f"{input_column}: {df.at[idx, input_column]}\n{output_column}:"
                                }
                            )
                        )
                    
                    # Get response
                    response = openai.chat.completions.create(
                        model=model,
                        messages=[message.to_dict() for message in tmp_messages]
                    )
                    answer = response.choices[0].message.content.strip()
                    tmp_messages.append(
                        Message(
                            **{
                                'role': response.choices[0].message.role,
                                'content': response.choices[0].message.content
                            }
                        )
                    )
                    df.at[idx, output_column] = answer.split(f"{output_column}: ")[-1] if f"{output_column}: " in answer else answer

                    # Log the assistant message
                    for idx, message in enumerate(tmp_messages):
                        __log : str = ""
                        for k, v in message.to_dict().items():
                            if k == "role":
                                __log += f"[{idx}] {v}: "
                            elif k == "content":
                                logger.info(__log + v)
                            else:
                                raise ValueError("Unexpected Error.")
                    
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
                file_name=f"{uploaded_file.name}_{output_column}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
