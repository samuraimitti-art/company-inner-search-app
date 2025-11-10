"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################

import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import constants as ct

def handle_user_input(user_input):
    """
    ユーザー入力を処理してAI回答を生成
    """
    import streamlit as st
    
    # ベクターストアが初期化されているかチェック
    if not hasattr(st.session_state, 'vectorstore') or st.session_state.vectorstore is None:
        return "申し訳ございませんが、現在システムの初期化が完了していないため、検索機能を利用できません。OpenAI APIキーが正しく設定されているかご確認ください。"
    
    try:
        vectorstore = st.session_state.vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": ct.RETRIEVER_TOP_K})

        # ダミーベクターストアの場合は簡単な応答を返す
        if hasattr(st.session_state.vectorstore, '__class__') and 'Dummy' in st.session_state.vectorstore.__class__.__name__:
            results = retriever.get_relevant_documents(user_input)
            
            answer_text = "### 🔍 検索結果（テストモード）\n\n"
            answer_text += "**参照ドキュメント:**\n"
            for idx, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "不明なソース")
                page = doc.metadata.get("page", None)
                chunk = doc.metadata.get("chunk", None)
                
                if source.endswith(".pdf"):
                    if page is not None:
                        answer_text += f"{idx}. 📄 **{os.path.basename(source)}** - {page + 1}ページ目\n"
                    else:
                        answer_text += f"{idx}. 📄 **{os.path.basename(source)}**\n"
                elif chunk is not None:
                    answer_text += f"{idx}. 📄 **{os.path.basename(source)}** - セクション{chunk + 1}\n"
                else:
                    answer_text += f"{idx}. 📄 **{os.path.basename(source)}**\n"
            
            answer_text += "\n---\n"
            answer_text += f"**テスト応答:** \n入力内容「{user_input}」を受け取りました。\n"
            answer_text += "実際のOpenAI APIキーを設定すると、本格的なAI検索機能を利用できます。\n"
            answer_text += "現在はテストモードで動作しています。"
            
            return answer_text
        
        # 実際のベクターストアの場合
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

        results = retriever.get_relevant_documents(user_input)

        answer_text = "### 🔍 参照ドキュメント\n\n"
        for idx, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "不明なソース")
            page = doc.metadata.get("page", None)
            chunk = doc.metadata.get("chunk", None)

            if source.endswith(".pdf"):
                if page is not None:
                    answer_text += f"{idx}. 📄 **{os.path.basename(source)}** - {page + 1}ページ目\n"
                else:
                    answer_text += f"{idx}. 📄 **{os.path.basename(source)}**\n"
            elif source.endswith(".docx"):
                if chunk is not None:
                    answer_text += f"{idx}. 📄 **{os.path.basename(source)}** - セクション{chunk + 1}\n"
                else:
                    answer_text += f"{idx}. 📄 **{os.path.basename(source)}**\n"
            elif source.endswith(".csv"):
                if chunk is not None:
                    answer_text += f"{idx}. 📊 **{os.path.basename(source)}** - データセクション{chunk + 1}\n"
                else:
                    answer_text += f"{idx}. 📊 **{os.path.basename(source)}**\n"
            elif source.endswith(".txt"):
                if chunk is not None:
                    answer_text += f"{idx}. 📝 **{os.path.basename(source)}** - セクション{chunk + 1}\n"
                else:
                    answer_text += f"{idx}. � **{os.path.basename(source)}**\n"
            else:
                answer_text += f"{idx}. 📄 **{os.path.basename(source)}**\n"

        answer_text += "\n---\n"

        # LLM回答
        response = qa_chain.run(user_input)
        answer_text += f"**AI回答:**\n{response}"

        return answer_text
        
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"


def get_error_message():
    """
    エラーメッセージを返す関数
    """
    return "初期化処理に失敗しました。システム管理者にお問い合わせください。"
