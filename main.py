"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
import streamlit as st
from initialize import initialize
import utils
import constants as ct

# ============================================================
# 初期設定
# ============================================================
try:
    initialize()
except Exception as e:
    st.error(ct.INITIALIZE_ERROR_MESSAGE)
    st.stop()

# ページ設定
st.set_page_config(
    page_title=ct.APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# タイトル
# ============================================================
st.title("社内情報特化型生成AI検索アプリ")

# ============================================================
# サイドバー
# ============================================================
st.sidebar.title("利用目的")

answer_mode = st.sidebar.radio(
    "利用目的を選択してください：",
    [ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
    index=0,
    key="answer_mode_radio"
)
st.session_state.answer_mode = answer_mode

st.sidebar.markdown("----")

if answer_mode == ct.ANSWER_MODE_1:
    st.sidebar.markdown("### 🗂「社内文書検索」を選択した場合")
    st.sidebar.info("入力内容と関連性が高い社内文書のありかを検索できます。")
    st.sidebar.caption("【入力例】社員の育成方針に関するMTGの議事録")
else:
    st.sidebar.markdown("### 💬「社内問い合わせ」を選択した場合")
    st.sidebar.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
    st.sidebar.caption("【入力例】人事部に所属している従業員情報を一覧化して")

# ============================================================
# メイン画面（右側）
# ============================================================

# グリーティングメッセージ（緑ボックス）
st.markdown(
    """
    <div style="background-color:#E8F5E9; padding:15px; border-radius:10px; margin-bottom:10px;">
        <b>👋 こんにちは。</b><br>
        私は、社内文書の情報をもとに回答する生成AIチャットボットです。<br>
        サイドバーで利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。
    </div>
    """,
    unsafe_allow_html=True
)

# 注意メッセージ（黄色ボックス）
st.markdown(
    """
    <div style="background-color:#FFF8E1; padding:15px; border-radius:10px; margin-bottom:20px;">
        ⚠️ <b>具体的に入力したほうが期待通りの回答を得やすいです。</b>
    </div>
    """,
    unsafe_allow_html=True
)

# チャット履歴
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 入力欄
user_input = st.chat_input("こちらからメッセージを送信してください。")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = utils.handle_user_input(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
