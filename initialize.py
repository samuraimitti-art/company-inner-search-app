"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################

import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, CSVLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import constants as ct

def create_dummy_vectorstore():
    """
    テスト用のダミーベクターストアを作成
    """
    from langchain.schema import Document
    
    # ダミードキュメントを作成
    dummy_docs = [
        Document(
            page_content="これはテスト用のドキュメントです。会社の基本情報や製品情報が含まれています。",
            metadata={"source": "test_doc.txt", "page": 0}
        ),
        Document(
            page_content="社内検索システムのテストデータです。実際のOpenAI APIキーを設定すると、本格的な検索機能を利用できます。",
            metadata={"source": "test_doc2.txt", "page": 1}
        )
    ]
    
    # ダミークラスでベクターストアをモック
    class DummyVectorStore:
        def as_retriever(self, **kwargs):
            return DummyRetriever(dummy_docs)
    
    class DummyRetriever:
        def __init__(self, docs):
            self.docs = docs
            
        def get_relevant_documents(self, query):
            # 常に最初の2つのドキュメントを返す
            return self.docs[:2]
    
    return DummyVectorStore()


def load_documents(data_path):
    documents = []
    
    # テキストスプリッターの初期化
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP
    )
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            # 拡張子ごとに処理
            if ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
                pdf_docs = loader.load()
                # チャンク分割を適用（メタデータを保持）
                split_docs = text_splitter.split_documents(pdf_docs)
                # ページ番号情報が保持されていることを確認
                for doc in split_docs:
                    if "page" not in doc.metadata and "source" in doc.metadata:
                        # ページ情報がない場合は0を設定
                        doc.metadata["page"] = 0
                documents.extend(split_docs)

            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
                docx_docs = loader.load()
                # チャンク分割を適用
                split_docs = text_splitter.split_documents(docx_docs)
                # DOCXファイルにはページ概念がないため、チャンク番号を追加
                for idx, doc in enumerate(split_docs):
                    doc.metadata["chunk"] = idx
                documents.extend(split_docs)

            elif ext == ".csv":
                loader = CSVLoader(file_path)
                csv_docs = loader.load()
                # チャンク分割を適用
                split_docs = text_splitter.split_documents(csv_docs)
                # CSVファイルには行番号情報を追加
                for idx, doc in enumerate(split_docs):
                    doc.metadata["chunk"] = idx
                documents.extend(split_docs)

            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                txt_docs = loader.load()
                # チャンク分割を適用
                split_docs = text_splitter.split_documents(txt_docs)
                # TXTファイルにはチャンク番号を追加
                for idx, doc in enumerate(split_docs):
                    doc.metadata["chunk"] = idx
                documents.extend(split_docs)

    return documents


def initialize():
    """
    アプリの初期化処理（安全版）
    """
    import streamlit as st
    from dotenv import load_dotenv
    
    # 環境変数の読み込み
    load_dotenv()
    
    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vectorstore" not in st.session_state:
        try:
            # OpenAI APIキーの確認
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key or openai_api_key.startswith("#") or "your-actual" in str(openai_api_key) or openai_api_key == "dummy_key_for_testing":
                # ダミーのベクターストアを作成してテスト機能を提供
                st.session_state.vectorstore = create_dummy_vectorstore()
                print("OpenAI APIキーが設定されていないため、テスト用のダミーデータを使用します。")
                return
            
            # データの読み込みとベクターストア作成
            data_path = "data"
            if os.path.exists(data_path):
                docs = load_documents(data_path)
                if docs:
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    st.session_state.vectorstore = vectorstore
                    print("ベクターストアが正常に作成されました。")
                else:
                    st.session_state.vectorstore = None
                    print("読み込み可能なドキュメントが見つかりません。")
            else:
                st.session_state.vectorstore = None
                print("dataフォルダが見つかりません。")
                
        except Exception as e:
            st.session_state.vectorstore = None
            print(f"初期化中にエラーが発生しました: {e}")
            raise e
