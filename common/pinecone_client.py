from pinecone import Pinecone
from .config import settings

pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index)
