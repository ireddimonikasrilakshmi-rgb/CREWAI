
# CrewAI Projects API

Backend API for 10 CrewAI projects using FastAPI and Gemini.

## Projects

1. Blog Post Crew
2. Competitor Analysis Crew
3. Meeting Notes Crew
4. Data Report Crew
5. Customer Support Flow
6. Resume Screening Flow
7. Bug Triage Flow
8. Content Repurposing Flow
9. Lead Qualification Flow
10. Approval Report Flow

## Requirements

- Python 3.10+
- FastAPI
- CrewAI
- Gemini API Key

## Environment Variables

Set the following environment variable:

GEMINI_API_KEY=your_gemini_api_key

Optional:

GEMINI_MODEL=gemini/gemini-3.1-flash-lite-preview

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Server

```bash
uvicorn app:app --reload
```

## API Documentation

Open:

/docs

## API Endpoints

All endpoints accept POST requests with JSON:

```json
{
  "input": "Your project input here"
}
```

Endpoints:

- POST /agent1
- POST /agent2
- POST /agent3
- POST /agent4
- POST /agent5
- POST /agent6
- POST /agent7
- POST /agent8
- POST /agent9
- POST /agent10

## Response

```json
{
  "output": "Generated result"
}
```

Project 10 also returns an approval status.

## Note

Keep your Gemini API key private. Never upload it to GitHub.
