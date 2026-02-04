UO RAG Agent.
The process is to get URLs, crawl web pages in parallel, store in Milvus, and run hybrid search in Milvus.
Each code file below includes a short description.

get_diffs_solutions/generate_url_list.py **This script crawls uoregon.edu and writes an XML url set with page hashes.**
get_diffs_solutions/generate_url_list_parallel.py **This script crawls uoregon.edu with threads and writes an XML url set with page hashes.**
get_diffs_solutions/thread_solution.py **This script recomputes page hashes and compares them to the saved XML using threads.**
get_diffs_solutions/get_crawl_diff.py **This script uses asyncio and aiohttp to compare current page hashes to the XML in parallel.**
local_solutions/crawl_uoregon.py **This script crawls pages with crawl4ai, chunks markdown, embeds text, and stores chunks in Postgres while initializing a Milvus collection.**
local_solutions/crawl_uoregon_milvus.py **This script crawls pages with crawl4ai, embeds chunks, and stores them in Milvus for hybrid search.**
local_solutions/keyword_search_milvus.py **This script runs BM25 keyword search against the Milvus sparse index.**
local_solutions/milvus_setup.py **This script creates a demo Milvus collection, inserts sample rows, and runs example queries.**
local_solutions/query.py **This script queries a few rows from the Milvus demo collection and prints vector info.**
local_solutions/local_ai_expert.py **This module defines a Pydantic AI agent that embeds queries with a local Ollama server and retrieves top Milvus chunks.**
local_solutions/streamlit_ui_local.py **This app provides a Streamlit chat UI that runs the Pydantic AI agent with Milvus dependencies.**
local_solutions/small_test.py **This script calls a local OpenAI compatible chat endpoint and prints a tool call example.**
