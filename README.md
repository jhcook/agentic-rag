# Agentic RAG

🚧 Project Under Construction 🚧

## Overview

Agentic RAG (Retrieval-Augmented Generation) is a project designed to enhance the capabilities of natural language processing by integrating retrieval mechanisms with generative models. This project allows users to index documents, perform searches, and synthesize answers based on retrieved information.

## Features

- **Document Indexing**: Index text documents for efficient retrieval.
- **Search Functionality**: Perform searches on indexed documents using natural language queries.
- **Answer Synthesis**: Generate answers based on retrieved documents.
- **Grounding Verification**: Verify the grounding of generated answers against the source documents.

## Installation

To get started with the Agentic RAG project, clone the repository and install the required Python packages:

```bash
$ git clone https://github.com/yourusername/agentic-rag.git
$ cd agentic-rag
agentic-rag$ pip install -r requirements.txt
...
```

The dependencies required are Ollama and mcphost:

```bash
$ brew install ollama
...
$ go install github.com/mark3labs/mcphost@latest
...
```

mcphost installed:

```bash
$ 
```

## Running

In different terminal windows:

```bash
$ ollama serve
```

```bash
agentic-rag$ python3 http_server.py
2025-10-21 10:07:36,790 - rag_core - INFO - Initialized new Store instance
2025-10-21 10:07:36,809 - __main__ - INFO - Loading document store...
2025-10-21 10:07:36,810 - rag_core - INFO - Loading store from ./rag_store.jsonl
2025-10-21 10:07:36,810 - rag_core - INFO - Initialized new Store instance
2025-10-21 10:07:36,810 - rag_core - INFO - Successfully loaded 5 documents
INFO:     Started server process [16353]
INFO:     Waiting for application startup.
2025-10-21 10:07:36,839 - mcp.server.streamable_http_manager - INFO - StreamableHTTP session manager started
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

```bash
agentic-rag$ 
```

```bash
agentic-rag$ mcphost --model ollama:llama3.2:1b --config mcp.yaml
```

## Development

Running tests:

```bash
agentic-rag$ pip install -r requirements-dev
...
agentic-rag$ pytest -v --cov=rag_core tests/ -s
...
```

Running the HTTP Server:

```bash
agentic-rag$ uv run python http_server.py
2025-10-21 09:39:20,482 - rag_core - INFO - Initialized new Store instance
2025-10-21 09:39:20,502 - rag_core - INFO - Loading store from ./rag_store.jsonl
2025-10-21 09:39:20,502 - rag_core - INFO - Initialized new Store instance
2025-10-21 09:39:20,502 - rag_core - INFO - Successfully loaded 5 documents
INFO:     Started server process [11508]
INFO:     Waiting for application startup.
2025-10-21 09:39:20,602 - mcp.server.streamable_http_manager - INFO - StreamableHTTP session manager started
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

REST API Server:

```bash
agentic-rag$ uvicorn rest_server:app --host 127.0.0.1 --port 8001
/Users/jcook/repo/agentic-rag/venv/lib/python3.14/site-packages/fastapi/_compat/v1.py:72: UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.
  from pydantic.v1 import BaseConfig as BaseConfig  # type: ignore[assignment]
2025-10-20 21:46:55,551 - rag_core - INFO - Initialized new Store instance
INFO:     Started server process [86841]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

MCP Server:

```bash
agentic-rag$ mcphost --model ollama:llama3.2:1b --config mcp.yaml

```

An example agent run:

```bash
agentic-rag$ ./agent.py "who is the nuix cto" ./rag
{
  "final": {
    "answer": "Draft based on 6 passages.\n\nQuestion: who is the nuix cto\n\nEvidence:\n[1] Nuix offers a suite of investigative analytics and eDiscovery products designed to help organizations extract insights from complex, unstructured data. Here's a breakdown of their key offerings:\n\nNuix Neo Platform\nA unified, AI-powered platform built around the patented Nuix Engine, enabling scalabl\n\n[2] Nuix Investigate is a collaborative, visual analytics tool designed to help investigators, legal teams, and compliance professionals uncover insights from complex digital evidence. It\u2019s part of the Nuix platform and built to accelerate case resolution by making data relationships easier to understan\n\n[3] Nuix Workstation is a powerful desktop application designed for forensic investigators, legal professionals, and compliance teams to process and analyze massive volumes of digital data. Here's a detailed breakdown:\n\nWhat It Does\nData Processing: Rapidly ingests and indexes structured, semi-structure\n\n[4] Jonathan Rubinsztein\nGlobal Chief Executive Officer\nJonathan has been the Group Chief Executive Officer since December 2021. He is a seasoned CEO with a track record of building world class global technology companies and leading high-performance teams in the technology sector. \n\nJonathan is a Non-E\n\n[5] Nuix Ltd (ASX: NXL) is an Australian technology company specializing in investigative analytics and intelligence software. Here's a breakdown of what Nuix does and why it's significant:\n\nWhat Nuix Offers\nCore Focus: Nuix helps organizations extract insights from vast amounts of unstructured data\u2014thi\n\n[6] Alexis Rouch CTO, GAICD Nuix\n\nAbout\nA senior executive with over twenty years experience in commercial and public sector organisations with a focus on providing inspirational and consistent leadership, working with stakeholders and developing staff to ensure long-term sustainable change.\n\nAlexis is \n\nAnswer (summarized; cite like [1], [2]): ...",
    "citations": [
      "rag/nuix_products.txt",
      "rag/nuix_investigate.txt",
      "rag/nuix_workstation.txt",
      "rag/global_ceo.txt",
      "rag/nuix.txt",
      "rag/cto.txt"
    ]
  },
  "verdict": {
    "answer_conf": 0.95,
    "citation_coverage": 1.0,
    "missing_facts": []
  },
  "iterations": 0
}
agentic-rag$ curl -s http://127.0.0.1:8001/api/search \
  -H 'content-type: application/json' \
  -d '{"query":"nuix CTO","k":1}' | jq
[
  {
    "text": "Alexis Rouch CTO, GAICD Nuix\n\nAbout\nA senior executive with over twenty years experience in commercial and public sector organisations with a focus on providing inspirational and consistent leadership, working with stakeholders and developing staff to ensure long-term sustainable change.\n\nAlexis is highly accomplished at managing technology and general operational areas and has held a variety of senior management positions predominantly in international banking and management consulting roles. From strategy development to successful delivery of significant change programmes, this has included direct line management of up to 600 employees as well as delivery through strategic partners and community organisations. \n\nBased in Australia, she has also spent considerable time working in the United Kingdom, Eastern Europe, India and South-East Asia. She has a track record as a consensus-builder and motivator across diverse cultural environments and has successfully developed relationships and influence at the most senior levels of organisations with board members, community leaders, politicians & government officials, customers and suppliers.\n\nRoles have included General Manager of Technology at WorkSafe, General Manager of Global Sourcing at ANZ, Executive Director at Lambeth Council (UK) and Director IT & Change Management at First National Bank (UK).\n",
    "score": 0.5,
    "uri": "rag/cto.txt",
    "meta": {}
  }
]
```

## Resources

* https://ollama.com/library/llama3.2
* https://docs.litellm.ai/docs/
* https://pypi.org/project/mcp/
* https://fastapi.tiangolo.com
* https://mcphub.tools/detail/mark3labs/mcphost
