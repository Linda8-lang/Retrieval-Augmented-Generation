**Business Education AI Assistant**  
This app allows users to ask business education questions and receive intelligent, context-aware answers by leveraging both preloaded and user uploaded documents. It uses document embeddings and semantic search to retrieve the most relevant context before generating a response using an LLM thus ollama via llama3.2:latest.

<!-- Load the knowledge base -->

* On app startup, the docs folder is scanned for PDFs.  
*  Each document is split into approximately 500 word chunks.  
* Sentence embeddings are generated using SentenceTransformers.  
* A FAISS index is created for fast similarity search.

<!-- Upload User Document (Optional) -->

* Users can upload one or more PDFs via Streamlit User Interface.  
* Uploaded documents are:  
  * Parsed using fitz.  
  * Chunked into about 500 word documents  
  * Embedded and indexed in-memory using FAISS.  
* The uploaded documents override the default docs during the current session.  
* Uploaded files are optionally saved to the uploaded\_docs folder.

<!-- Ask a question -->

* User enters a question into the text box.  
* The app determines whether to search.  
  * The default index (if no uploads)  
  * The uploaded index (if uploads exist)

<!-- Retrieve Relevant context -->

* The question is embedded using the same model.  
* FAISS is used to find the top-k most similar chunks.  
* The chunks are combined to form a context.

<!-- Construct prompt & Query LLM -->

* A prompt is constructed using “As a Business Education teacher. Use the context below to answer the user's question.”  
  * Context:{retrieved chunks} 
  * Question:{user query}  
* The  prompt is sent to the locally running ollama API(llama3.2:latest)  
* A response is returned and displayed.

<!-- Manage uploads -->

* User can: Upload multiple files in one go.  
  *     Click the bin to clear uploaded files to reset the session

<!-- Requirements -->

* Streamlit: Defines the user interactions with the app and how the app behaves.  
* PyMuPDF(fitz): Extracting text from the large document body.  
* SentenceTransformers: Embedding neural network model: all-MiniLM-L6-v2  
* Ollama(llama3.2:latest): Local LLM response generation.

 <!-- How to Use -->

1. Start Ollama locally and pull the model:  
``` 
ollama pull llama3.2  
ollama run llama3.2  
```

2. Install requirements:  
```
pip install -r requirements.txt  
```

3. Run Streamlit app:  
``` 
streamlit run bust_assistant.py  
```

