# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import datetime

def main():
    st.markdown("# 1. データをアップロードします")
    uploaded_file = st.file_uploader("CSVファイルをドラッグ&ドロップ、またはブラウザから選択してください", type='csv', key='train')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("# 2. アップロードされたデータを確認します")
        st.dataframe(df.head(10))

        st.markdown("# 3. ターゲットを入力してください")
        target = st.text_input(label='ターゲット名を文字列で正しく入力してください', value=df.columns[-1])

        st.markdown("# 4. 回帰の場合はregression、分類の場合はclassificationを選択してください")
        ml_usecase = st.selectbox(label='ドロップダウンリストからregressionかclassificationを選択してください',
                                options=('', 'regression', 'classification'),
                                key='ml_usecase')
        if ml_usecase == 'regression':
            from pycaret.regression import *
        elif ml_usecase == 'classification':
            from pycaret.classification import *
        else:
            st.text('選択せれていません')  

        if (ml_usecase == 'regression') | (ml_usecase == 'classification'):

            st.markdown("# 5. 実行します")
            st.markdown("実行中です…しばらくお待ち下さい…")
            ml = setup(data=df,
                    target=target,
                    session_id=1234,
                    silent=True,
            )

            best = compare_models()
            st.dataframe(best)
            select_model = best.index[0]
            model = create_model(select_model)
            final = finalize_model(model)
            save_model(final, select_model+'_saved_'+datetime.date.today().strftime('%Y%m%d'))
            st.markdown("モデル構築が完了しました")              
    
if __name__ == '__main__':
    main()