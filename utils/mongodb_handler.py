"""MongoDB handler for storing CUA interactions."""
import os
from datetime import datetime
from typing import Dict, List, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

class MongoDBHandler:
    """Handles all MongoDB operations for the CUA system."""
    
    def __init__(self):
        self.uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = os.getenv("MONGODB_DB", "cua_pdf_reader")
        self.client = None
        self.db = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Force connection
            self.db = self.client[self.db_name]
            print(f"✅ Connected to MongoDB: {self.db_name}")
        except ConnectionFailure as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise
    
    def store_interaction(self, interaction_data: Dict) -> str:
        """Store a user interaction."""
        collection = self.db.interactions
        interaction_data["timestamp"] = datetime.utcnow()
        result = collection.insert_one(interaction_data)
        return str(result.inserted_id)
    
    def store_extracted_content(self, content_data: Dict) -> str:
        """Store extracted content from VLM."""
        collection = self.db.extracted_content
        content_data["timestamp"] = datetime.utcnow()
        result = collection.insert_one(content_data)
        return str(result.inserted_id)
    
    def store_question_answer(self, qa_data: Dict) -> str:
        """Store question and answer pair."""
        collection = self.db.qa_pairs
        qa_data["timestamp"] = datetime.utcnow()
        result = collection.insert_one(qa_data)
        return str(result.inserted_id)
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict]:
        """Get recent interactions."""
        collection = self.db.interactions
        cursor = collection.find().sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    def get_paper_history(self, paper_id: str) -> List[Dict]:
        """Get all interactions for a specific paper."""
        collection = self.db.interactions
        cursor = collection.find({"paper_id": paper_id})
        return list(cursor)
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

# Test the connection
if __name__ == "__main__":
    handler = MongoDBHandler()
    print("MongoDB handler initialized successfully!")
    handler.close()