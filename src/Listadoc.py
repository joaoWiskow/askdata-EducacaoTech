"""Lista todos os arquivos indexados no ChromaDB, com a contagem de chunks de cada um."""
import chromadb
from collections import Counter
 
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="askdata_knowledge",
    metadata={"hnsw:space": "cosine"}
)
 
resultado = collection.get(include=["metadatas"])
arquivos = [m["arquivo"] for m in resultado["metadatas"]]
 
contagem = Counter(arquivos)
 
print(f"Total de chunks na base: {len(arquivos)}")
print(f"Total de documentos distintos: {len(contagem)}\n")
for arquivo, qtd in sorted(contagem.items()):
    print(f"  - {arquivo}: {qtd} chunks")
 