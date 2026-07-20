import json
import logging
import time
from typing import Any, Dict, Union

import logfire
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


@logfire.instrument("node:supervisor_plan", extract_args=False)
async def ainvoke_supervisor_plan_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[str, list, dict]]:
    """Supervisor node that generates a structured literature survey plan.

    Pauses the graph using interrupt_before on the next node for Human-in-the-Loop review.

    :param state: Current agent state
    :param runtime: Runtime context containing LLM details
    :returns: Updated state dictionary with the generated research plan
    """
    logger.info("NODE: supervisor_plan")
    messages = state["messages"]
    query = state.get("sanitized_query") or get_latest_query(messages)

    prompt = (
        "You are an Academic Director. Your task is to design a structured, professional "
        "literature review plan consisting of exactly 3 core sections to investigate the following topic:\n\n"
        f"Topic: {query}\n\n"
        "Define exactly 3 sections. For each section, provide a title, a detailed research description, "
        "and exactly 2 highly specific technical search queries optimized for finding relevant papers on arXiv.\n\n"
        "Output your outline strictly as a JSON object matching this schema:\n"
        "{\n"
        '  "sections": [\n'
        "    {\n"
        '      "title": "Section Title",\n'
        '      "description": "What this section will research and synthesize",\n'
        '      "queries": ["query 1", "query 2"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Write only the raw JSON. Do not include markdown code block formatting (like ```json or ```)."
    )

    try:
        model = runtime.context.llm_client.get_langchain_model(runtime.context.model_name, temperature=0.0)
        response = await model.ainvoke(prompt)
        content = response.content.strip()

        # Handle potential markdown code blocks if the LLM ignored instructions
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        plan_data = json.loads(content)
        sections = plan_data.get("sections", [])

        logger.info(f"✓ Supervisor generated outline plan with {len(sections)} sections")
        return {
            "research_plan": sections,
            "original_query": query,
            "messages": [
                AIMessage(content=f"Supervisor has generated the research outline plan:\n{json.dumps(sections, indent=2)}")
            ],
        }
    except Exception as e:
        logger.error(f"Failed to generate structured plan: {e}. Falling back to default plan.")

        # Safe fallback plan structure
        fallback_plan = [
            {
                "title": "Foundational Concepts & Background",
                "description": f"Core mathematical models, origins, and architectures of {query}.",
                "queries": [f"{query} architecture", f"{query} foundations"],
            },
            {
                "title": "Core Methodology & Technical Mechanics",
                "description": f"Deep dive into the operational algorithms and innovations of {query}.",
                "queries": [f"{query} algorithms", f"state-of-the-art {query}"],
            },
            {
                "title": "Empirical Evaluations & Future Directions",
                "description": f"Performance benchmarks, comparisons, limitations, and future challenges of {query}.",
                "queries": [f"{query} evaluation benchmarks", f"{query} limitations challenges"],
            },
        ]
        return {
            "research_plan": fallback_plan,
            "original_query": query,
            "messages": [AIMessage(content="Supervisor generated fallback outline plan.")],
        }
