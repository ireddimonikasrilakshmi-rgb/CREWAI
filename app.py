
import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM

app = FastAPI(
    title="CrewAI Projects API",
    description="Backend API for 10 CrewAI projects",
    version="1.0.0",
)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini/gemini-3.1-flash-lite-preview")


class Query(BaseModel):
    input: str = Field(..., min_length=1, description="Text input for the project")


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server."
        )
    return LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.7,
        max_tokens=1500,
    )


async def run_single_agent(
    user_input: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    expected_output: str,
) -> str:
    llm = get_llm()
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=False,
    )
    task = Task(
        description=task_description.format(user_input=user_input),
        expected_output=expected_output,
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    result = await crew.kickoff_async()
    return str(result)


async def run_multi_agent(user_input: str, specs: List[dict]) -> str:
    llm = get_llm()
    agents = []
    tasks = []
    for spec in specs:
        agent = Agent(
            role=spec["role"],
            goal=spec["goal"],
            backstory=spec["backstory"],
            llm=llm,
            verbose=False,
        )
        agents.append(agent)
        tasks.append(Task(
            description=spec["task"].format(user_input=user_input),
            expected_output=spec["expected"],
            agent=agent,
        ))
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    result = await crew.kickoff_async()
    return str(result)


@app.get("/")
def home():
    return {
        "message": "CrewAI Projects API is running",
        "docs": "/docs",
        "routes": [f"/agent{i}" for i in range(1, 11)],
    }


# Project 1: Blog Post Crew
@app.post("/agent1")
async def blog_post(q: Query):
    try:
        output = await run_multi_agent(q.input, [
            {
                "role": "Blog Researcher",
                "goal": "Research the requested blog topic and identify useful information.",
                "backstory": "You are a careful researcher who prepares accurate topic notes.",
                "task": "Research this blog topic: {user_input}. Provide key facts, useful points, benefits, challenges, and examples.",
                "expected": "A clear research summary."
            },
            {
                "role": "Blog Writer and Editor",
                "goal": "Write and polish an informative blog post.",
                "backstory": "You are a professional blog writer and editor.",
                "task": "Using the research context, write a well-structured blog post about: {user_input}. Include a title, introduction, headings, and conclusion. Check clarity and grammar.",
                "expected": "A polished blog post."
            },
        ])
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project 2: Competitor Analysis Crew
@app.post("/agent2")
async def competitor_analysis(q: Query):
    try:
        output = await run_single_agent(
            q.input, "Competitor Strategy Analyst",
            "Compare competitors and prepare a SWOT analysis.",
            "You are a business analyst. Do not invent facts, prices, or sources.",
            """Analyze the competitors and information supplied here: {user_input}.
            Compare products, services, positioning and pricing only where information is supplied.
            Clearly mark missing information. Finish with a SWOT analysis for the comparison.""",
            "A competitor comparison and SWOT report."
        )
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project 3: Meeting Notes Crew
@app.post("/agent3")
async def meeting_notes(q: Query):
    try:
        output = await run_multi_agent(q.input, [
            {
                "role": "Meeting Summarizer",
                "goal": "Summarize the meeting transcript.",
                "backstory": "You create concise and accurate meeting summaries.",
                "task": "Summarize this meeting transcript: {user_input}",
                "expected": "A concise meeting summary."
            },
            {
                "role": "Action Item Finder",
                "goal": "Extract tasks, owners and deadlines.",
                "backstory": "You carefully identify action items from meetings.",
                "task": "From this meeting transcript, identify action items, responsible people and deadlines: {user_input}",
                "expected": "A list of action items with owners and deadlines."
            },
            {
                "role": "Meeting Notes Formatter",
                "goal": "Prepare organized meeting notes.",
                "backstory": "You format professional meeting records.",
                "task": "Prepare final meeting notes from the transcript: {user_input}. Include summary, action items, decisions and next steps.",
                "expected": "Formatted meeting notes."
            },
        ])
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project 4: Data Report Crew
@app.post("/agent4")
async def data_report(q: Query):
    try:
        output = await run_multi_agent(q.input, [
            {
                "role": "Data Analyst",
                "goal": "Analyze the supplied data and identify patterns.",
                "backstory": "You are a careful data analyst. Do not invent data values.",
                "task": "Analyze the following data or dataset description: {user_input}. Calculate totals and comparisons when the values are provided, and identify trends.",
                "expected": "A factual data analysis with calculations where possible."
            },
            {
                "role": "Report Writer",
                "goal": "Write a clear data report.",
                "backstory": "You prepare professional business reports.",
                "task": "Turn the supplied data and analysis into a report: {user_input}. Include introduction, findings, performance and conclusion.",
                "expected": "A structured data report."
            },
        ])
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project 5: Customer Support Flow
@app.post("/agent5")
async def customer_support(q: Query):
    try:
        output = await run_single_agent(
            q.input, "Customer Support Classifier",
            "Classify a customer complaint and prepare an appropriate response.",
            "You are an experienced customer support specialist.",
            """Read this customer complaint: {user_input}
            First classify it as exactly one of billing, account_access, or other.
            Billing and account_access are High priority; other is Normal priority.
            Then provide a polite and useful first response. Clearly show Category, Priority and Response.""",
            "Category, priority and customer response."
        )
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project 6: Resume Screening Flow
@app.post("/agent6")
async def resume_screening(q: Query):
    try:
        output = await run_single_agent(
            q.input, "Resume Evaluator",
            "Evaluate a resume for a software developer role.",
            "You are an experienced HR recruiter. Evaluate only the information supplied.",
            """Evaluate the resume and job requirements supplied here: {user_input}
            Give a score from 0 to 10, explain the reason, identify strengths and skill gaps,
            and classify as SHORTLIST if score is 7 or above, otherwise REJECT.
            Do not infer personal characteristics unrelated to job skills.""",
            "Candidate name if supplied, score out of 10, reason, strengths, gaps and shortlist/reject decision."
        )
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project 7: Bug Triage Flow
@app.post("/agent7")
async def bug_triage(q: Query):
    try:
        output = await run_single_agent(
            q.input, "Issue Triage Analyst",
            "Classify a software issue and recommend the next action.",
            "You are a software support analyst.",
            """Read this issue: {user_input}
            Classify it as exactly one of BUG, FEATURE, or QUESTION.
            For BUG, explain likely investigation steps; for FEATURE, suggest implementation considerations;
            for QUESTION, provide a clear answer. Show the category first.""",
            "Issue category and a useful next action or answer."
        )
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project 8: Content Repurposing Flow
@app.post("/agent8")
async def content_repurposing(q: Query):
    try:
        output = await run_single_agent(
            q.input, "Content Repurposing Specialist",
            "Turn one article into content for multiple platforms.",
            "You are a social media and newsletter content specialist.",
            """Repurpose this article or source text: {user_input}
            First extract its key points. Then create:
            1. A professional LinkedIn post.
            2. Three concise Twitter/X posts.
            3. A short newsletter with a subject and body.
            Preserve the original meaning and do not add unsupported facts.""",
            "Key points, LinkedIn post, three Twitter/X posts and newsletter."
        )
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project 9: Lead Qualification Flow
@app.post("/agent9")
async def lead_qualification(q: Query):
    try:
        output = await run_single_agent(
            q.input, "Lead Qualification Specialist",
            "Classify a sales lead as hot, warm or cold and recommend follow-up.",
            "You are a sales operations analyst.",
            """Evaluate this lead information: {user_input}
            Classify the lead as HOT, WARM, or COLD based only on supplied information.
            Explain the reason and prepare a suitable follow-up message.
            Do not claim that a website was visited unless website content is included.""",
            "Lead category, explanation and follow-up message."
        )
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Project 10: Approval Report Flow
@app.post("/agent10")
async def approval_report(q: Query):
    try:
        output = await run_multi_agent(q.input, [
            {
                "role": "Research Analyst",
                "goal": "Prepare research findings for a report.",
                "backstory": "You are a careful research analyst. Identify uncertain claims.",
                "task": "Research and organize findings for this report topic and audience: {user_input}",
                "expected": "Organized report research and findings."
            },
            {
                "role": "Report Writer",
                "goal": "Draft a professional report for human review.",
                "backstory": "You write clear, structured reports.",
                "task": "Draft a report based on the topic and research request: {user_input}. Include title, introduction, main sections, conclusion and references where available. Mark it clearly as DRAFT FOR HUMAN APPROVAL.",
                "expected": "A complete draft report awaiting human approval."
            },
        ])
        return {
            "output": output,
            "status": "awaiting_human_approval",
            "note": "Review this draft before publishing. This API endpoint does not itself record a human approval."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
