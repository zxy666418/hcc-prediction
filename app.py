import streamlit as st
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. 助手函数 (来自 demo.py)
# ----------------------------------------------------------------------

def _friendly_feature_names(names):
    """(来自 demo.py) 把 one-hot 后的特征名转为易读名字"""
    pretty_names_map = {}
    for n in names:
        if n.startswith("cat__"):
            base = n[len("cat__"):].split('_')[0]
            pretty_names_map[n] = base
        elif n.startswith("num__"):
            pretty_names_map[n] = n[len("num__"):]
        else:
            pretty_names_map[n] = n
    return [pretty_names_map[n] for n in names]


# (!!! 关键修复：重写 SHAP 绘图函数 !!!)
def generate_shap_plot(shap_values_single, base_value,
                       input_df_row,
                       ohe_feature_names):
    """
    (!!! 最终修复版 !!!)
    此版本通过“手动聚合”来解决 SHAP 库的 Bug。
    它将 23 个 OHE/标准化 后的 SHAP 值，
    手动聚合成 20 个原始特征的 SHAP 值。
    """

    # --- 1. 我们需要 3 个新列表 (长度都为 20) ---
    agg_shap_values = []  # 聚合后的 SHAP 值
    agg_data_values = []  # 原始输入值
    agg_feature_names = []  # 20 个唯一特征名

    # --- 2. 创建一个字典来聚合 SHAP 值 ---
    # e.g., {'AFP': 0.92, 'Surgery2': 0.0, ...}
    agg_shap_map = {}

    # --- 3. 遍历 23 个 OHE 后的 SHAP 值 ---
    for ohe_name, shap_val in zip(ohe_feature_names, shap_values_single):
        # 将 'cat__Surgery2_1.0' 转换为 'Surgery2'
        agg_name = _friendly_feature_names([ohe_name])[0]

        # (!!! 聚合步骤 !!!)
        # 将 OHE 特征的 SHAP 值加在一起
        if agg_name not in agg_shap_map:
            agg_shap_map[agg_name] = shap_val
        else:
            agg_shap_map[agg_name] += shap_val

    # --- 4. 按 20 个原始特征的顺序，构建最终列表 ---
    for agg_name in ALL_FEATURES_ORDERED:
        # 1. 添加特征名
        agg_feature_names.append(agg_name)

        # 2. 添加聚合后的 SHAP 值
        agg_shap_values.append(agg_shap_map.get(agg_name, 0.0))

        # 3. 添加对应的 *原始* 输入值 (e.g., 100.5)
        agg_data_values.append(input_df_row[agg_name])

    # --- 5. 创建一个干净的 (20行) Explanation 对象 ---
    exp = shap.Explanation(
        values=np.array(agg_shap_values),  # 20 个聚合后的 SHAP 值
        base_values=base_value,
        data=np.array(agg_data_values),  # 20 个原始输入值
        feature_names=agg_feature_names  # 20 个唯一特征名
    )

    # --- 6. 绘图 (并移除所有方框) ---
    fig, ax = plt.subplots(figsize=(9, 6))

    # max_display=20 将显示所有 20 个特征
    shap.plots.waterfall(exp, max_display=11, show=False)

    for txt in ax.texts:
        txt.set_bbox(None)
    # for patch in ax.patches:
    #     if 'Polygon' not in str(type(patch)):
    #         patch.set_visible(False)

    print(f"[SHAP Plot] 瀑布图生成成功，已手动聚合 OHE 特征。")

    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2. 加载模型 (使用缓存)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """加载所有新生成的、仅20个特征的模型文件"""
    print("Loading models...")
    model = joblib.load('best_hcc_recurrence_model.pkl')
    preprocessor = joblib.load('preprocess.pkl')
    vt = joblib.load('variance_threshold.pkl')
    selected_features_names = joblib.load('selected_features_after_preprocess.pkl')
    selected_feature_idx = joblib.load('selected_feature_idx.npy')

    explainer = shap.TreeExplainer(model, feature_perturbation = "tree_path_dependent", model_output = "raw")

    return model, preprocessor, vt, selected_features_names, selected_feature_idx, explainer


# 加载所有文件
model, preprocessor, vt, \
    selected_features_names, selected_feature_idx, \
    explainer = load_artifacts()

# ----------------------------------------------------------------------
# 3. 定义20个特征的列表 (用于表单和 DataFrame)
# ----------------------------------------------------------------------
NUMERIC_FEATURES = [
    'AFP', 'RET', 'MCV', 'NER', 'WBC', 'GLU', 'MPV', 'EOS%',
    'PT', 'GLR', 'GGT', 'BUN', 'Time', 'MMR'
]
CATEGORICAL_FEATURES = [
    'Surgery2', 'Tumor capsule', 'Liver cirrhosis',
    'Gallbladder invasion', 'Major vascular invasion', 'Tumor number'
]
# 这个列表用于构建 DataFrame 和 聚合 SHAP
ALL_FEATURES_ORDERED = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ----------------------------------------------------------------------
# 4. Streamlit 页面状态管理 和 样式注入
# ----------------------------------------------------------------------
st.set_page_config(layout="centered", page_title="HCC Prediction")

# (!!! CSS：灰色背景 + 白色卡片 + 蓝色按钮 !!!)
st.markdown(
    """
    <style>
    /* 页面整体容器样式 */
    [data-testid="stBlockContainer"] {
        background-color: #f5f5f5;  
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* 预测按钮样式（蓝色） */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #0066cc; 
        color: white; 
        border: none;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #004a99; 
        color: white;
        border: none;
    }

    /* 标题样式 */
    h1 {
        color: #0066cc;
    }

    /* 子标题样式 */
    h3 {
        color: #333333;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 5px;
    }

    /* Go Back按钮样式（蓝色） */
    button[onclick="go_back()"] {
        background-color: #0066cc !important;
        color: white !important;
        border: none !important;
    }
    button[onclick="go_back()"]:hover {
        background-color: #004a99 !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if 'show_results' not in st.session_state:
    st.session_state.show_results = False


def go_to_results():
    st.session_state.show_results = True


def go_back():
    if 'prediction_label' in st.session_state:
        del st.session_state.prediction_label
    if 'shap_fig' in st.session_state:
        del st.session_state.shap_fig
    st.session_state.show_results = False





# ----------------------------------------------------------------------
# 5. 渲染页面
# ----------------------------------------------------------------------

if st.session_state.show_results:

    # ========================
    #  页面 2: 结果页
    # ========================
    st.markdown(f'<h1 style="text-align: center;">Prediction Result</h1>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background-color: #ffebee; border: 1px solid #e57373; border-radius: 8px; padding: 20px; text-align: center;">
            <h2 style="color: #c62828; margin: 0;">{st.session_state.prediction_label}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(f'<h3>Prediction Explanation (SHAP Waterfall Plot)</h3>', unsafe_allow_html=True)
    st.pyplot(st.session_state.shap_fig)
    st.button("Go Back", on_click=go_back, type="primary",use_container_width=True)

else:

    # ========================
    #  页面 1: 输入表单
    # ========================
    st.markdown(f'<h1 style="text-align: center;">HCC Postoperative Early Recurrence Prediction</h1>',
                unsafe_allow_html=True)

    default_inputs = st.session_state.get('form_inputs', {})

    with st.form("input_form"):
        form_inputs = {}
        st.subheader("Numeric Features")

        col1, col2, col3 = st.columns(3)
        with col1:
            form_inputs['AFP'] = st.number_input("AFP（ng/ml）", value=0.0, format="%.3f")
            form_inputs['RET'] = st.number_input("RET（×10⁹/L）", value=0.0, format="%.3f")
            form_inputs['MCV'] = st.number_input("MCV（fl）", value=0.0, format="%.3f")
            form_inputs['NER'] = st.number_input("NER", value=0.0, format="%.3f")
            form_inputs['WBC'] = st.number_input("WBC（×10⁹/L）", value=0.0, format="%.3f")
        with col2:
            form_inputs['GLU'] = st.number_input("GLU（mmol/L）", value=0.0, format="%.3f")
            form_inputs['MPV'] = st.number_input("MPV（fl）", value=0.0, format="%.3f")
            form_inputs['EOS%'] = st.number_input("EOS%（%）", value=0.0, format="%.3f")
            form_inputs['PT'] = st.number_input("PT（s）", value=0.0, format="%.3f")
            form_inputs['GLR'] = st.number_input("GLR", value=0.0, format="%.3f")
        with col3:
            form_inputs['GGT'] = st.number_input("GGT(U/L)", value=0.0, format="%.3f")
            form_inputs['BUN'] = st.number_input("BUN(mmol/L)", value=0.0, format="%.3f")
            form_inputs['Time'] = st.number_input("Time (min)", value=0.0, format="%.3f")
            form_inputs['MMR'] = st.number_input("MMR（cm）", value=0.0, format="%.3f")

        st.markdown("---")
        st.subheader("Categorical Features")

        col1, col2, col3 = st.columns(3)

        options_binary = [0, 1]
        options_surgery2 = [1, 2, 3, 4]
        options_tumor_num = [1, 2]


        def get_index(options, default_key):
            val = default_inputs.get(default_key, options[0])
            try:
                return options.index(val)
            except ValueError:
                return 0  # 如果保存的值无效，返回第一个


        with col1:
            form_inputs['Surgery2'] = st.selectbox("Surgery2", options_surgery2,
                                                   index=get_index(options_surgery2, 'Surgery2'))
            form_inputs['Tumor capsule'] = st.selectbox("Tumor capsule", options_binary,
                                                        index=get_index(options_binary, 'Tumor capsule'))
        with col2:
            form_inputs['Liver cirrhosis'] = st.selectbox("Liver cirrhosis", options_binary,
                                                          index=get_index(options_binary, 'Liver cirrhosis'))
            form_inputs['Gallbladder invasion'] = st.selectbox("Gallbladder invasion", options_binary,
                                                               index=get_index(options_binary, 'Gallbladder invasion'))
        with col3:
            form_inputs['Major vascular invasion'] = st.selectbox("Major vascular invasion", options_binary,
                                                                  index=get_index(options_binary,
                                                                                  'Major vascular invasion'))
            form_inputs['Tumor number'] = st.selectbox("Tumor number", options_tumor_num,
                                                       index=get_index(options_tumor_num, 'Tumor number'))

        st.markdown("---")

        submitted = st.form_submit_button("Predict", use_container_width=True)

        if submitted:
            try:
                input_df = pd.DataFrame([form_inputs])[ALL_FEATURES_ORDERED]
                st.session_state.form_inputs = form_inputs

                X_proc = preprocessor.transform(input_df)
                X_vt = vt.transform(X_proc)
                X_selected = X_vt[:, selected_feature_idx]

                proba = model.predict_proba(X_selected)[0, 1]
                shap_values = explainer.shap_values(X_selected)
                base_value = explainer.expected_value

                # (!!! 关键修复：调用新的聚合函数 !!!)
                fig = generate_shap_plot(
                    shap_values_single=shap_values[0],
                    base_value=base_value,
                    input_df_row=input_df.iloc[0],  # 20 个原始值
                    ohe_feature_names=selected_features_names  # 23 个 OHE 名字
                )

                st.session_state.prediction_label = f"High Risk ({proba * 100:.2f}%)" if proba > 0.5 else f"Low Risk ({proba * 100:.2f}%)"
                st.session_state.shap_fig = fig

                go_to_results()
                st.rerun()

            except Exception as e:
                st.error(f"预测时发生错误: {e}")
                st.error("请确保您的 .pkl 文件与 20 个特征列表完全对应。")