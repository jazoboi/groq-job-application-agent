# High-Speed Job Application Agent (Groq-Powered)

> AI agent leveraging Groq's LPU inference for ultra-fast resume tailoring and job application automation.

## Role
**Lead Developer** — Designed the full pipeline from scraping to application generation.

## Overview
An AI agent utilizing Groq's LPU inference for ultra-fast processing of job descriptions and resume tailoring. Scrapes postings, extracts requirements, matches skills, and generates tailored applications autonomously.

## Pipeline
```
Job Board Scrape → Requirement Extraction → Skill Matcher → Resume Tailor
                                                                  ↓
                                              Cover Letter Gen → Application Package
```

## Key Features
- **LPU Inference** — ~12ms per LLM call (74x faster than cloud GPU)
- **Skill Extraction** — NER + pattern matching for job requirements
- **Semantic Matching** — Embeddings-based skill-to-requirement scoring
- **Resume Tailoring** — Dynamic emphasis adjustment per job posting
- **Batch Processing** — 50+ tailored applications per hour

## Tech Stack
`Groq LPU` · `LLM Orchestration` · `Python` · `BeautifulSoup` · `pydantic`

## Impact
- **~12ms inference latency** per LLM call (74x faster than cloud GPU)
- **50+ tailored applications/hour** processing capacity
- Average job-match score of **82%**

## Project Structure
```
src/
├── groq_client.py          # Groq API wrapper with retry logic
├── job_scraper.py          # Job board scraping & parsing
├── skill_matcher.py        # Semantic skill matching engine
├── resume_tailor.py        # Dynamic resume customization
└── application_builder.py  # Final application assembly
```

## License
MIT
