# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tempfile

def convert_df(df):
   return df.to_csv(index=False).encode('shift-jis')

st.cache_resource()
def select_model(ml_model,):
    model = create_model(ml_model)  # モデルを作成
    return model

st.markdown("# AutoML Tool")

st.sidebar.markdown("データを読み込みます")
uploaded_new_file = st.sidebar.file_uploader("CSVファイルをドラッグ&ドロップ、またはブラウザから選択してください", type='csv', key='train')

if uploaded_new_file is not None:

    df_new = pd.read_csv(uploaded_new_file, encoding="shift-jis")
    st.markdown("#### 1. アップロードされたデータを確認下さい")
    st.dataframe(df_new.head(10))

    st.markdown("#### 2. 目的変数を入力してください")
    st.markdown("##### 目的変数以外の変数で推定を実施します")
    target = st.selectbox(label='目的変数を指定してください', options=df_new.keys())

    st.markdown("### 3. 分析手法を選択ください")
    ml_usecase = st.selectbox(label='ドロップダウンリストからregressionかclassificationを選択してください',
                            options=('', 'regression', 'classification'),
                            key='ml_usecase')
    if ml_usecase == 'regression':
        from pycaret.regression import *
    elif ml_usecase == 'classification':
        from pycaret.classification import *
    else:
        st.text('選択されていません')  

    if (ml_usecase == 'regression') | (ml_usecase == 'classification'):

        if 'execute_flg' not in st.session_state:
            st.session_state["execute_flg"] = 0

        if st.session_state["execute_flg"] == 0:
            st.markdown("#### 実行中です…しばらくお待ち下さい…")
            
            ml = setup(data=df_new,
                    target=target,
                    session_id=1234,
                    ignore_features = target[0],
            )

            best = compare_models()
        
        st.write('#### Metrics')
        st.dataframe(pull(), height=200)
        #st.write(best)
                    
        st.markdown("#### モデル構築が完了しました")
        st.session_state["execute_flg"] = 1

        st.markdown("### 4.モデルを選択してください")
        ml_model = st.selectbox(label='モデルを選択してください',
                                options=('','dt', 'gbr', 'par', 'et', 'rf', 'knn', 'ada', 'lightgbm', 'lr',\
                                        'lar', 'omp', 'br', 'ridge', 'en', 'huber', 'dummy', 'llar', 'lasso'),
                                key='ml_model')
        if ml_model != '':
            st.text(ml_model)
        else:
            st.text('選択されていません')  

        if (ml_model != ''):
            if 'select_model_flg' not in st.session_state:
                st.session_state["select_model_flg"] = 0

            #if st.session_state["select_model_flg"] == 0:
            model = select_model(ml_model)
            
            # Plots
            plot_model(model, plot='residuals', display_format='streamlit')
            plot_model(model, plot='feature', display_format='streamlit')
            plot_model(model, plot='error', display_format='streamlit')
            
            st.markdown("#### モデルのFineTuneが完了しました")
            st.session_state["select_model_flg"] = 1

            pred_holdout = predict_model(model)
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

            # 一時的なファイルを作成してモデルを保存
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                pickle.dump(model, temp_file, protocol=4)
                temp_file_path = temp_file.name

            # モデルのダウンロード
            with open(temp_file_path, 'rb') as file:
                model_binary = file.read()

            st.download_button(
            label="Download Model",
            data=open(temp_file_path, "rb").read(),
            file_name="model.pkl",
            key="download-model-button",
            help="Click to download the trained model.",
        )