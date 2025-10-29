# RAG-Based-AI-Teaching-Assistant

## Purpose
The goal of this project is to build an AI-powered Teaching Assistant for the Data Science course. The assistant can understand students’ questions, retrieve the most relevant information from course materials, and generate accurate, context-aware answers — just like a real teaching assistant.

This system is particularly useful for:
->Students who want quick explanations or clarifications from lectures or notes
->Educators who want to automate question-answering based on their own materials



## How It Works

1)Speech-to-Text Conversion (Whisper)

->The project uses OpenAI Whisper (whisper.load_model("base")) to transcribe MP3 lecture files into text.

->The output is stored as a JSON file containing time-aligned text segments for further use.

2)Knowledge Base Preparation

->The transcribed lecture text (and optionally notes or PDFs) is cleaned and indexed.

->Each chunk of content is converted into embeddings (vector representations) and stored in a vector database.

3)Retrieval-Augmented Generation (RAG)

->When a user asks a question, the system retrieves the most relevant content from the vector store using similarity search.

->The retrieved context is passed to the LLaMA 3.2 model, which generates a context-aware response based on both the query and the course materials.

4)Final Output

->The assistant responds to the user with an accurate, human-like explanation grounded in the course’s actual data science content.
