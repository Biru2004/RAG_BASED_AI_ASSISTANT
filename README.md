# RAG_BASED_AI_ASSISTANT
Project Description

The RAG-based AI Teaching Assistant is an advanced Retrieval-Augmented Generation (RAG) system designed to transform video-based educational content into an interactive, searchable knowledge base. By leveraging state-of-the-art Natural Language Processing (NLP) and Computer Vision utilities, the assistant allows users to query specific information contained within large video datasets. The system bypasses the need for manual scrubbing through hours of footage by transcribing, indexing, and retrieving precise segments of information to generate context-aware, accurate responses.
Technical Implementation & Workflow

The architecture begins with a robust Data Ingestion layer. Using FFmpeg, the system programmatically processes raw video files to extract high-quality audio streams. These streams are then fed into OpenAI’s Whisper, a world-class Speech-to-Text model, which generates highly accurate transcripts. To handle the computational limits of Large Language Models (LLMs), the project utilizes LangChain for sophisticated text chunking and orchestration. These text chunks are converted into high-dimensional numerical vectors using specialized Embedding Models.

To ensure efficient information retrieval, the system stores these vectors in a Vector Database (such as FAISS or ChromaDB). For version control of large-scale binary data, such as the embeddings.joblib file, Git LFS (Large File Storage) is integrated into the workflow, ensuring that high-performance datasets are managed seamlessly within the GitHub ecosystem. When a user submits a query, the assistant performs a similarity search to find the most relevant context, which is then synthesized by an LLM to provide a coherent, teaching-oriented answer.
Key Technologies Used

    Programming Language: Python

    Audio/Video Engineering: FFmpeg (Processing and Stream Extraction)

    Speech Recognition: OpenAI Whisper (Transcription)

    Framework Orchestration: LangChain (RAG Pipeline & Prompt Engineering)

    Vector Embeddings: HuggingFace Transformers / OpenAI Embeddings

    Storage & Versioning: Git LFS (for managing 90MB+ .joblib embedding files)

    Data Management: Pandas & NumPy (Preprocessing)

    Model Persistence: Joblib (Efficient serialization of vector data)
