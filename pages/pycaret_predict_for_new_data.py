# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import shutil
import tempfile

def convert_df(df):
   return df.to_csv(index=False).encode('shift-jis')

def predict(loaded_model, df):
    pred_holdout = predict_model(loaded_model, data=df)
    
    plot_model(loaded_model, plot='residuals', display_format='streamlit')
    plot_model(loaded_model, plot='feature', display_format='streamlit')
    plot_model(loaded_model, plot='error', display_format='streamlit')

    st.markdown("#### output the result as prediction_label")
    st.dataframe(pred_holdout, height=200)
    csv = convert_df(pred_holdout)
    st.download_button(
            "Download result",
            csv,
            "predict.csv",
            "text/csv",
            key='download-csv'
            )
    return pred_holdout

st.markdown("# AutoML Tool")

#st.sidebar.markdown("import the data")
uploaded_file = st.sidebar.file_uploader("import the data", type='csv', key='train')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="shift-jis")
    st.markdown("#### 1. check the data")
    st.dataframe(df.head(10))

    st.markdown("### 2. select the method")
    ml_usecase = st.selectbox(label='regression or classification',
                            options=('', 'regression', 'classification'),
                            key='ml_usecase')
    if ml_usecase == 'regression':
        from pycaret.regression import *
    elif ml_usecase == 'classification':
        from pycaret.classification import *
    
    
    target = st.selectbox(label='input objective variable', options=df.keys())
    pkl_file = st.file_uploader("import model", type=".pkl", accept_multiple_files=False)

    if pkl_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        temp_file.write(pkl_file.read())
        temp_file.close()
        
        # 一時的なファイルの拡張子を".pkl"に修正
        temp_file_path_with_extension = temp_file.name + ".pkl"
        shutil.move(temp_file.name, temp_file_path_with_extension)

        loaded_model = load_model(temp_file.name)
        ml = setup(data=df,
                    target=target,
            )
            
        predict(loaded_model, df)
