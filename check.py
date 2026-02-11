import requests
CHROMA_HOST = "localhost"  # или имя сервиса Docker
CHROMA_PORT = 8000

try:
    r = requests.get(f"http://{CHROMA_HOST}:{CHROMA_PORT}/health")
    if r.status_code == 200 and r.json().get("ok"):
        print("ChromaDB доступна")
    else:
        print("ChromaDB недоступна")
except Exception as e:
    print("Ошибка подключения к ChromaDB:", e)