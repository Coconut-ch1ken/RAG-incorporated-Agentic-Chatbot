import os
import sys
from dotenv import load_dotenv
from src.ingestion.csv_loader import CSVIngestor
from src.database.vector_store import VectorStore
from src.database.metadata_store import MetadataStore
from src.graph.workflow import RagAgent

# Load environment variables
load_dotenv()

def main():
    print("Initialize RAG Agent...")
    
    # Initialize components
    # We share the store instances
    v_store = VectorStore()
    m_store = MetadataStore()
    ingestor = CSVIngestor(v_store, m_store)
    agent = RagAgent()
    
    # Hardcoded user for demo
    USER_ID = "demo_user"
    
    print(f"RAG Agent initialized for User: {USER_ID}")
    print("Commands:")
    print("  /ingest <path_to_csv>  - Upload a CSV file")
    print("  /exit                  - Quit")
    print("  <any text>             - Chat with your data")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == "/exit":
                print("Goodbye!")
                break
                
            if user_input.lower().startswith("/ingest "):
                file_path = user_input[8:].strip()
                if not os.path.exists(file_path):
                    print(f"Error: File '{file_path}' not found.")
                    continue
                
                print(f"Ingesting {file_path}...")
                ingestor.ingest(file_path, USER_ID)
                continue
            
            # Chat
            print("Thinking...")
            result = agent.run(user_input, USER_ID)
            
            # Parse result
            if "generation" in result:
                print(f"\nAgent: {result['generation']}")
                if "hallucination_status" in result and result['hallucination_status']:
                    print("✅ Verified grounded")
                else: 
                     # If we exited due to no data
                    if not result.get("documents"):
                        print("❌ No relevant data found in your files.")
                    else:
                        print("⚠️ Answer may not be fully grounded.")
            else:
                 print("Error: No generation produced.")
                 
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
