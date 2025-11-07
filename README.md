MLN131 RAG Chatbot (FastAPI + openai)

Hướng dẫn nhanh
- Cấu trúc thư mục đã tạo:
  - app/: mã nguồn FastAPI, RAG, Gemini client
  - config/settings.json: cấu hình chunking, model, đường dẫn
  - scripts/build_index.py: script tạo FAISS index từ data
  - storage/: lưu index và docstore
  - data/data.txt: dữ liệu nguồn
  - final_model/: model embedding (SentenceTransformers) đã fine-tune

Thiết lập môi trường
python 3.10
