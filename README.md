***UO RAG Agent***

This project implements a RAG pipeline that crawls the University of Oregon website, chunks the data, and stores it in a Milvus vector database. This information is used in an agentic RAG system to query information about the University of Oregon.

This pipline is split into 3 different scripts. 
  1) get URLs from (generate_url_list_parallel.py)
  2) crawl web pages in parallel and store in Milvus (crawl_uoregon_milvus.py)
  3) run hybrid search in Milvus and return data to open source Ollama model (streamlit_ui_local.py)

***File Descriptions***

**generate_url_list_parallel.py**: This script crawls uoregon.edu with threads and writes an XML url set with page hashes.

**get_diffs_solutions/get_crawl_diff.py**: This script uses asyncio and aiohttp to compare current page hashes to the XML in parallel.

**crawl_uoregon_milvus.py**: This script crawls pages with crawl4ai, embeds chunks, and stores them in Milvus for hybrid search.

**milvus_setup.py**: This script creates a demo Milvus collection, inserts sample rows, and runs example queries.

**local_ai_expert.py**: This module defines a Pydantic AI agent that embeds queries with a local Ollama server and retrieves top Milvus chunks.

**streamlit_ui_local.py**: This app provides a Streamlit chat UI that runs the Pydantic AI agent with Milvus dependencies.


***Additional scripts for testing***

**local_solutions/query.py**: This script queries a few rows from the Milvus demo collection and prints vector info.

**local_solutions/small_test.py**: This script calls a local OpenAI compatible chat endpoint and prints a tool call example.

**local_solutions/keyword_search_milvus.py**: This script runs BM25 keyword search against the Milvus sparse index.
