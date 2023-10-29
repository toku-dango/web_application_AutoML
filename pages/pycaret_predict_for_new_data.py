# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd

def convert_df(df):
   return df.to_csv(index=False).encode('shift-jis')

def predict(loaded_model, df):
    pred_holdout = predict_model(loaded_model, data=df)
    
    plot_model(loaded_model, plot='residuals', display_format='streamlit')
    plot_model(loaded_model, plot='feature', display_format='streamlit')
    plot_model(loaded_model, plot='error', display_format='streamlit')

    st.markdown("#### 推定結果がprediction_labelに出力されます")
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

@st.cache(allow_output_mutation=True)
def create_model_cache(estimator):
    return create_model(estimator)

st.markdown("# AutoML Tool")

st.sidebar.markdown("推定に使用するデータを読み込みます")
uploaded_file = st.sidebar.file_uploader("CSVファイルをドラッグ&ドロップ、またはブラウザから選択してください", type='csv', key='train')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="shift-jis")
    st.markdown("#### 1. アップロードされたデータを確認下さい")
    st.dataframe(df.head(10))

    st.markdown("### 2. 分析手法を選択ください")
    ml_usecase = st.selectbox(label='ドロップダウンリストからregressionかclassificationを選択してください',
                            options=('', 'regression', 'classification'),
                            key='ml_usecase')
    if ml_usecase == 'regression':
        from pycaret.regression import *
    elif ml_usecase == 'classification':
        from pycaret.classification import *

    # pkl_lt = [''] + [f[:-4] for f in os.listdir(os.getcwd()) if f[-4:]=='.pkl']
    # pkl_file = st.selectbox(label='ドロップダウンリストからモデルを選択してください',
    #                           options=pkl_lt,
    #                           key='model')

    pkl_file = st.text_input("#### ファイル名を入力下さい")
    if pkl_file != (''):
        loaded_model = load_model(pkl_file)
        
        target = st.selectbox(label='目的変数を指定してください', options=df.keys())
        ml = setup(data=df,
                    target=target,
            )
        predict(loaded_model, df)

    # model_usecase = st.selectbox(label='モデルをアップロードするか、作成したものを使用するのか選択ください',
    #                         options=('', 'アップロード', '直近で作成したものを使用'),
    #                         key='model_usecase')
    # if model_usecase == ('アップロード'):
    #     uploaded_model = st.sidebar.file_uploader("モデルをアップロードください")
        
    #     if uploaded_model is not None:
    #         loaded_model = load_model(uploaded_model)
    #         predict(loaded_model, df)

    # elif model_usecase == ('直近で作成したものを使用'):
    #     loaded_model = load_model('saved_model_'+datetime.date.today().strftime('%Y%m%d'))
    #     predict(loaded_model, df)
