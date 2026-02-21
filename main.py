import os
import sys
from dotenv import load_dotenv
from src.database.vector_store import VectorStore
from src.database.metadata_store import MetadataStore
from src.ingestion.directory_scanner import DirectoryScanner
from src.graph.workflow import RagAgent
from src.config import settings

# Load environment variables
load_dotenv()


def main():
    print("🤖 Initializing Personal Assistant RAG Agent...")

    # Initialize components
    v_store = VectorStore()
    m_store = MetadataStore()
    scanner = DirectoryScanner(v_store, m_store)
    agent = RagAgent()

    # Hardcoded user for demo
    USER_ID = "demo_user"

    print(f"Agent initialized for User: {USER_ID}")
    print(f"Default data directory: {settings.watch_directory}")
    print(f"Local LLM: {settings.ollama_model} | Embeddings: {settings.ollama_embed_model}")
    print()
    print("Commands:")
    print("  /scan                  - Scan default data directory")
    print("  /scan <path>           - Scan a specific directory")
    print("  /files                 - List ingested files")
    print("  /exit                  - Quit")
    print("  <any text>             - Chat with your data")

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/exit":
                print("Goodbye! 👋")
                break

            # --- /scan command ---
            if user_input.lower().startswith("/scan"):
                parts = user_input.split(maxsplit=1)
                scan_dir = parts[1].strip() if len(parts) > 1 else settings.watch_directory

                if not os.path.isdir(os.path.expanduser(scan_dir)):
                    print(f"Error: Directory '{scan_dir}' not found.")
                    continue

                print(f"📂 Scanning directory: {scan_dir}")
                stats = scanner.scan(scan_dir, USER_ID)
                print(f"\n📊 Scan complete — "
                      f"Ingested: {stats['ingested']} | "
                      f"Skipped: {stats['skipped']} | "
                      f"Errors: {stats['errors']}")
                continue

            # --- /files command ---
            if user_input.lower() == "/files":
                files = m_store.get_user_files(USER_ID)
                if files:
                    print("📁 Ingested files:")
                    for f in files:
                        print(f"  • {f}")
                else:
                    print("No files ingested yet. Use /scan to ingest data.")
                continue

            # --- Chat ---
            print("🔍 Thinking...")
            result = agent.run(user_input, USER_ID)

            # Parse result
            if "generation" in result and result["generation"]:
                tier = result.get("generation_tier", "unknown")
                tier_labels = {"local": "🏠 Local", "local+gemini": "🏠+☁️ Local+Gemini", "powerful": "💪 Powerful", "gemini": "☁️  Gemini"}
                tier_label = tier_labels.get(tier, f"❓ {tier}")

                print(f"\nAssistant: {result['generation']}")

                if result.get("hallucination_status"):
                    print(f"  ✅ Verified grounded ({tier_label})")
                else:
                    if not result.get("documents"):
                        print("  ❌ No relevant data found in your files.")
                    else:
                        print(f"  ⚠️  Answer may not be fully grounded ({tier_label})")

                # Show source citations
                sources = result.get("sources", [])
                if sources:
                    print(f"  📎 Sources: {', '.join(sources)}")
            else:
                if not result.get("documents"):
                    print("❌ No relevant data found. Try ingesting more files with /scan.")
                else:
                    print("Error: No generation produced.")

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
