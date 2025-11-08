import json
import os
import time
from pathlib import Path
from typing import Optional
import re
import argparse
import sys
import subprocess

def ensure_dependencies(config: dict | None = None):
    """Đảm bảo các package bắt buộc đã có. Nếu thiếu sẽ tự động cài bằng pip.

    Cài gói theo từng module để tránh thất bại toàn bộ khi một package (ví dụ faiss-cpu) không khả dụng.
    """
    required = {
        "fastapi": "fastapi",
        "pydantic": "pydantic",
        "dotenv": "python-dotenv",
        "uvicorn": "uvicorn",
        "sentence_transformers": "sentence-transformers",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "numpy": "numpy",
        "torch": "torch",
    }

    use_faiss = False
    try:
        use_faiss = bool(config.get("use_faiss", False)) if config else False
    except Exception:
        use_faiss = False

    def _try_import(modname: str) -> bool:
        try:
            __import__(modname)
            return True
        except Exception:
            return False

    def _install(pkg: str):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError:
            pass

    for mod, pkg in required.items():
        if not _try_import(mod):
            _install(pkg)

    if use_faiss and (not _try_import("faiss")):
        _install("faiss-cpu")

    critical = ["fastapi", "pydantic", "uvicorn", "sentence_transformers", "sklearn", "numpy"]
    missing = [m for m in critical if not _try_import(m)]
    if missing:
        raise RuntimeError(
            "Thiếu các thư viện bắt buộc: " + ", ".join(missing) +
            "\nVui lòng chạy: pip install -r requirements.txt hoặc để chương trình tự cài đặt có kết nối Internet."
        )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_config() -> dict:
    """Đọc config/settings.json luôn theo đường dẫn tuyệt đối của project root."""
    src_dir = Path(__file__).resolve().parent
    cfg_path = src_dir / "config" / "settings.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

ensure_dependencies(config)

import uvicorn
import logging
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Union

# Cấu hình logging: chỉ hiển thị thông tin cần thiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Tắt các log không cần thiết từ các thư viện
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from src.rag_service import RagService
from src.clients.openai_client import OpenAIClient
from src.clients.ollama_client import OllamaClient
from src.utils.chunking import chunk_text
from src.utils.preprocess import preprocess_text

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="MLN131 RAG Chatbot", version="2.0.0")

# Cấu hình CORS để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (có thể thay bằng danh sách cụ thể trong production)
    allow_credentials=False,  # Phải False khi dùng allow_origins=["*"]
    allow_methods=["*"],  # Cho phép tất cả methods
    allow_headers=["*"],  # Cho phép tất cả headers
)

rag: RagService = RagService(config)
ollama_client: Optional[OllamaClient] = None
openai_client: Optional[OpenAIClient] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    image_urls: Optional[List[str]] = None
    file_urls: Optional[List[str]] = None
    use_websearch: Optional[bool] = False


class RebuildRequest(BaseModel):
    backend: Optional[str] = None


@app.on_event("startup")
def startup_event():
    global ollama_client, openai_client
    logger.info("=" * 60)
    logger.info("[STARTUP] Đang khởi tạo các clients...")
    
    response_language = config.get("response_language", "vi")
    max_output_tokens = int(config.get("max_output_tokens", 150))
    temperature = float(config.get("temperature", 0.2))
    
    # Khởi tạo GPTOSS 20B Finetune client (cho câu hỏi text và vision)
    base_url = os.getenv("OLLAMA_BASE_URL")
    if not base_url:
        raise ValueError("OLLAMA_BASE_URL environment variable is required")
    ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "gpt-oss:20b")
    vision_model_name = os.getenv("OLLAMA_VISION_MODEL_NAME", "gemma3:latest")
    logger.info(f"[STARTUP] Khởi tạo GPTOSS 20B Finetune client: {ollama_model_name} tại {base_url}")
    ollama_client = OllamaClient(
        base_url=base_url,
        model_name=ollama_model_name,
        vision_model_name=vision_model_name,
        response_language=response_language,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    logger.info(f"[STARTUP] ✓ GPTOSS 20B Finetune client đã sẵn sàng")
    
    # Khởi tạo OpenAI client (cho câu hỏi có ảnh)
    api_key = os.getenv("OPENAI_API_KEY", "")
    openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano")
    if not api_key:
        raise ValueError("OPENAI_API_KEY không được tìm thấy trong environment variables")
    logger.info(f"[STARTUP] Khởi tạo OpenAI client: {openai_model_name}")
    openai_client = OpenAIClient(
        api_key=api_key,
        model_name=openai_model_name,
        response_language=response_language,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    
    logger.info(f"[STARTUP] ✓ OpenAI client đã sẵn sàng")
    
    # Load RAG index
    logger.info("[STARTUP] Đang load RAG index...")
    try:
        rag.load_index()
        logger.info("[STARTUP] ✓ RAG index đã load thành công")
    except Exception as e:
        logger.warning(f"[STARTUP] Không thể load index hiện có: {e}, đang build lại...")
        project_root = Path(__file__).resolve().parent.parent
        data_path_cfg = config.get("data_path", "data/data.txt")
        data_path = (project_root / data_path_cfg) if not Path(data_path_cfg).is_absolute() else Path(data_path_cfg)
        if not data_path.exists():
            raise FileNotFoundError(f"Không thấy file dữ liệu: {data_path}")
        logger.info(f"[STARTUP] Đang đọc và xử lý dữ liệu từ {data_path}")
        text = data_path.read_text(encoding="utf-8")
        text = preprocess_text(text)
        chunks = chunk_text(
            text,
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 120)),
            separators=config.get("separators", None),
            source=str(data_path)
        )
        logger.info(f"[STARTUP] Đã tạo {len(chunks)} chunks, đang build index...")
        rag.build_index(chunks)
        rag.load_index()
        logger.info("[STARTUP] ✓ Đã build và load RAG index thành công")
    
    logger.info("=" * 60)
    logger.info("[STARTUP] ✓ Tất cả services đã sẵn sàng!")
    logger.info("=" * 60)


@app.get("/health")
def health():
    chunk_count = 0
    try:
        chunk_count = len(rag.docstore)
    except Exception:
        pass
    return {"status": "ok", "index_ready": rag.is_ready(), "chunk_count": chunk_count}


def _is_about_vivi(question: str) -> bool:
    """Kiểm tra xem câu hỏi có liên quan đến vivi (cấu hình chatbot) không."""
    question_lower = question.lower()
    keywords = [
        "vivi", 
        "cấu hình", "cấu hình chatbot", 
        "chatbot của bạn", 
        "bạn là ai", "who are you", "what is your name",
        "giới thiệu về bạn", "tell me about",
        "bot này", "bot tên", "tên gì", "tên bạn",
        "bạn làm gì", "what do you do"
    ]
    return any(keyword in question_lower for keyword in keywords)


# Đã xóa hàm _needs_web_search - websearch giờ được bật thủ công qua nút toggle


def _get_bot_config_info() -> str:
    """Trả về thông tin cấu hình chatbot."""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "N/A")
    ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "gpt-oss:20b")
    openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano")
    
    info_parts = [
        "👋 Xin chào! Tôi là ViVi, một chatbot hỗ trợ thông tin dựa trên RAG (Retrieval-Augmented Generation).",
        "",

        "- **Tôi được tạo ra:** Để tự động chọn model phù hợp dựa trên loại câu hỏi của bạn",

    ]
    
    if openai_model_name.startswith("gpt-4.1-nano"):
        info_parts.extend([
            "- Hỗ trợ web search (OpenAI)",
            "- Hỗ trợ xử lý hình ảnh và file (OpenAI)",
        ])
    
    info_parts.append("")
    info_parts.append("💡 Bạn có thể hỏi tôi bất kỳ câu hỏi nào liên quan đến dữ liệu đã được lưu trữ!")
    
    return "\n".join(info_parts)


@app.post("/query")
def query(req: QueryRequest):
    start = time.perf_counter()
    logger.info(f"[QUERY] Nhận câu hỏi: {req.question[:100]}...")
    
    # Kiểm tra nếu câu hỏi về vivi thì trả về thông tin cấu hình
    if _is_about_vivi(req.question):
        logger.info("[QUERY] Câu hỏi về bot config, trả về thông tin cấu hình")
        answer = _get_bot_config_info()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return {
            "question": req.question,
            "answer": answer,
            "contexts": [],
            "meta": {"type": "bot_info"},
            "latency_ms": elapsed_ms
        }
    
    contexts_for_llm = []
    
    # RAG Search
    logger.info(f"[RAG] Bắt đầu search với top_k={req.top_k or rag.top_k_default}")
    results = rag.search(req.question, top_k=req.top_k)
    logger.info(f"[RAG] Tìm thấy {len(results)} kết quả")
    
    # Filter theo similarity threshold
    similarity_threshold = float(config.get("similarity_threshold", 0.6))
    filtered = [r for r in results if float(r.get("score", 0.0)) >= similarity_threshold]
    logger.info(f"[RAG] Sau filter (threshold={similarity_threshold}): {len(filtered)} contexts")
    
    # Giới hạn số lượng contexts
    contexts_max = int(config.get("contexts_max", 3))
    contexts_for_llm = filtered[:contexts_max]
    if contexts_for_llm:
        scores_str = ', '.join([f'{c["score"]:.3f}' for c in contexts_for_llm])
        logger.info(f"[RAG] Sử dụng {len(contexts_for_llm)} contexts (scores: [{scores_str}])")
    else:
        logger.warning("[RAG] Không có contexts nào đạt ngưỡng similarity")
    
    # Websearch được bật thủ công qua nút toggle từ frontend
    use_websearch = req.use_websearch or False
    
    # Xử lý: có ảnh HOẶC bật web search → GPT-4.1 nano (OpenAI), không có → GPTOSS 20B Finetune
    image_urls = req.image_urls or []
    file_urls = req.file_urls or []
    
    # Validate image formats - OpenAI chỉ hỗ trợ: png, jpeg, gif, webp
    validated_image_urls = []
    allowed_mime_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp']
    
    for img_url in image_urls:
        if isinstance(img_url, str) and img_url.startswith('data:'):
            # Extract MIME type from data URL
            mime_match = img_url.split(';')[0].split(':')[1] if ':' in img_url else None
            if mime_match and mime_match in allowed_mime_types:
                validated_image_urls.append(img_url)
            elif mime_match == 'image/svg+xml':
                return JSONResponse(
                    status_code=400,
                    content={"error": "Định dạng SVG không được hỗ trợ. Vui lòng sử dụng định dạng: png, jpeg, gif, hoặc webp."}
                )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Định dạng hình ảnh không được hỗ trợ: {mime_match}. Chỉ hỗ trợ: png, jpeg, gif, webp."}
                )
        else:
            # Nếu không phải data URL, giả sử hợp lệ (có thể là URL)
            validated_image_urls.append(img_url)
    
    image_urls = validated_image_urls
    has_images = len(image_urls) > 0 or len(file_urls) > 0
    
    # Chọn model dựa trên điều kiện
    if has_images or use_websearch:
        logger.info(f"[MODEL] Chọn OpenAI ({openai_client.model_name}) - has_images={has_images}, use_websearch={use_websearch}")
        # Có ảnh hoặc bật web search → gọi GPT-4.1 nano (OpenAI)
        answer, meta = openai_client.answer(
            req.question,
            contexts_for_llm,
            image_urls=image_urls if image_urls else None,
            file_urls=file_urls if file_urls else None,
            use_websearch=use_websearch
        )
    else:
        logger.info(f"[MODEL] Chọn GPTOSS 20B Finetune ({ollama_client.model_name}) - không có ảnh và không websearch")
        # Không có ảnh và không cần web search → dùng GPTOSS 20B Finetune
        answer, meta = ollama_client.answer(
            req.question, 
            contexts_for_llm
        )
    
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(f"[QUERY] Hoàn thành trong {elapsed_ms}ms, độ dài answer: {len(answer)} chars")
    
    return {
        "question": req.question,
        "answer": answer,
        "contexts": contexts_for_llm,
        "meta": meta,
        "latency_ms": elapsed_ms
    }


def _wc(s: str) -> int:
    return len(re.findall(r"\w+", s))


@app.get("/chunks")
def chunks(limit: int = 3, preview_chars: int = 300):
    """Xem nhanh các chunk đã build (preview)."""
    limit = max(1, min(limit, 50))
    pcs = []
    for i, c in enumerate(rag.docstore[:limit]):
        txt = c.get("text", "")
        pcs.append({
            "id": i,
            "source": c.get("source", "unknown"),
            "word_count": _wc(txt),
            "preview": txt[:preview_chars]
        })
    return {"chunk_count": len(rag.docstore), "preview_count": len(pcs), "chunks": pcs}


@app.post("/admin/rebuild_index")
def rebuild_index(req: RebuildRequest):
    global rag
    cfg = load_config()
    if req.backend:
        cfg["backend"] = req.backend
    project_root = Path(__file__).resolve().parent.parent
    data_path_cfg = cfg.get("data_path", "data/data.txt")
    data_path = (project_root / data_path_cfg) if not Path(data_path_cfg).is_absolute() else Path(data_path_cfg)
    text = data_path.read_text(encoding="utf-8")
    text = preprocess_text(text)
    chunks = chunk_text(
        text,
        chunk_size=int(cfg.get("chunk_size", 800)),
        chunk_overlap=int(cfg.get("chunk_overlap", 120)),
        separators=cfg.get("separators", None),
        source=str(data_path)
    )
    new_rag = RagService(cfg)
    new_rag.build_index(chunks)
    new_rag.load_index()
    rag = new_rag
    return {"status": "rebuilt", "backend": cfg.get("backend"), "index_ready": rag.is_ready(), "chunks": len(chunks)}


@app.post("/query/upload")
async def query_with_upload(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None),
    top_k: Optional[int] = Form(None),
    use_websearch: Optional[bool] = Form(False)
):
    """Query với hỗ trợ upload file (text/ảnh)."""
    start = time.perf_counter()
    
    # Kiểm tra nếu câu hỏi về vivi thì trả về thông tin cấu hình
    if _is_about_vivi(question):
        answer = _get_bot_config_info()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return {
            "question": question,
            "answer": answer,
            "contexts": [],
            "meta": {"type": "bot_info"},
            "latency_ms": elapsed_ms
        }
    
    contexts_for_llm = []
    image_urls = []
    
    if file:
        content = await file.read()
        file_ext = file.filename.split('.')[-1].lower() if file.filename else ''
        
        if file_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            # Encode ảnh thành base64 để gửi đến OpenAI API
            import base64
            base64_image = base64.b64encode(content).decode('utf-8')
            mime_type = f"image/{file_ext}" if file_ext != 'jpg' else "image/jpeg"
            image_data_url = f"data:{mime_type};base64,{base64_image}"
            image_urls.append(image_data_url)
        else:
            try:
                text_content = content.decode('utf-8')
                processed = preprocess_text(text_content)
                file_chunks = chunk_text(
                    processed,
                    chunk_size=int(config.get("chunk_size", 800)),
                    chunk_overlap=int(config.get("chunk_overlap", 120)),
                    source=f"uploaded:{file.filename}"
                )
                contexts_for_llm.extend([{"text": c["text"], "source": c["source"], "score": 1.0} for c in file_chunks[:2]])
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Không thể đọc file: {str(e)}"}
                )
    
    results = rag.search(question, top_k=top_k)
    similarity_threshold = float(config.get("similarity_threshold", 0.6))
    filtered = [r for r in results if float(r.get("score", 0.0)) >= similarity_threshold]
    contexts_max = int(config.get("contexts_max", 3))
    contexts_for_llm.extend(filtered[:contexts_max])
    
    # Websearch được bật thủ công qua nút toggle từ frontend
    use_websearch = use_websearch or False
    
    # Xử lý: có ảnh HOẶC bật web search → GPT-4.1 nano (OpenAI), không có → GPTOSS 20B Finetune
    has_images = len(image_urls) > 0
    
    if has_images or use_websearch:
        # Có ảnh hoặc bật web search → gọi GPT-4.1 nano (OpenAI)
        answer, meta = openai_client.answer(
            question,
            contexts_for_llm,
            image_urls=image_urls if image_urls else None,
            file_urls=None,
            use_websearch=use_websearch
        )
    else:
        # Không có ảnh và không cần web search → dùng GPTOSS 20B Finetune
        answer, meta = ollama_client.answer(
            question, 
            contexts_for_llm
        )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts_for_llm,
        "meta": meta,
        "latency_ms": elapsed_ms
    }


def _set_runtime_env_for_mac():
    """Thiết lập biến môi trường để server ổn định."""
    os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")
    os.environ.setdefault("TORCH_MPS_ENABLED", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


def _parse_args():
    parser = argparse.ArgumentParser(description="Chạy MLN131 FastAPI server")
    parser.add_argument("--host", default="127.0.0.1", help="Host (mặc định 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (mặc định 8000)")
    parser.add_argument("--reload", action="store_true", help="Bật reload khi phát triển")
    return parser.parse_args()


if __name__ == "__main__":
    _set_runtime_env_for_mac()
    args = _parse_args()
    if args.reload:
        uvicorn.run("src.main:app", host=args.host, port=args.port, reload=True, log_level="info")
    else:
        uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="info")
