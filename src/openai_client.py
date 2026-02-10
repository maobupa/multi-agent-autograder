"""
Shared OpenAI client utility to avoid duplicate initialization.
"""
import os
from dotenv import load_dotenv
import openai


class OpenAIClient:
    """Shared OpenAI client to avoid duplicate initialization across classes."""
    
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAIClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
            
            try:
                self._client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                raise ValueError(f"Failed to initialize OpenAI client: {e}. Please check your API key and OpenAI library version.")
    
    @property
    def client(self):
        return self._client
    
    def get_client(self):
        """Get the OpenAI client instance."""
        return self._client
