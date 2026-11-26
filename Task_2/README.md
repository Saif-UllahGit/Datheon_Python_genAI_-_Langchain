# Task 2: RAG-Based PDF Question Answering System

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions about their own PDF and text documents.

The system extracts text from uploaded documents, divides it into smaller chunks, generates vector embeddings, and stores them in a vector database. When a user asks a question, the system retrieves the most relevant information and generates an answer based strictly on the provided documents.

## Objectives

- Load PDF and text documents.
- Extract and process document content.
- Split text into smaller chunks using RecursiveCharacterTextSplitter.
- Generate vector embeddings.
- Store embeddings in ChromaDB.
- Retrieve relevant information based on user queries.
- Generate
