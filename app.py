import streamlit as st
import pandas as pd
import openai
from random import sample
import numpy as np
from io import BytesIO

# 엑셀 파일 업로드
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

def replace_semicolon_with_comma(df):
    # 모든 셀에 대해 문자열 변환을 시도합니다.
    return df.map(lambda x: str(x).replace(';', ',') if pd.notnull(x) else x)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    # df = convert_columns_to_arrow_compatible(df)  # Arrow 호환을 위한 자동 수정 적용
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
                examples = df[df[output_column].notna()].sample(10)
                prompt = "\n".join([f"{input_column}: {row[input_column]}, {output_column}: {row[output_column]}" for _, row in examples.iterrows()])
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
                    model="gpt-4o",
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
