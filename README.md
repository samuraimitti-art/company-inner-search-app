# 社内情報特化型生成AI検索アプリ

社内文書の情報をもとに回答する生成AIチャットボットアプリです。

## 機能

- 🗂 **社内文書検索**: 入力内容と関連性が高い社内文書を検索
- 💬 **社内問い合わせ**: 質問・要望に対して社内文書の情報をもとに回答

## デプロイ方法

### Streamlit Community Cloud

このアプリは Streamlit Community Cloud でデプロイできます。

#### 必要な環境変数

Streamlit Cloud の Settings → Secrets に以下を追加:

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

## ローカルでの実行

```bash
pip install -r requirements.txt
streamlit run main.py
```

## 技術スタック

- Streamlit
- LangChain
- OpenAI API
- FAISS
