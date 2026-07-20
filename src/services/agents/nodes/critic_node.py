import json
import logging
from typing import Dict, Union

import logfire
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState

logger = logging.getLogger(__name__)


@logfire.instrument("node:critic", extract_args=False)
async def ainvoke_critic_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[str, dict]]:
    """Critic / Peer Reviewer node.

    Evaluates the quality, citation integrity, and factuality of the written drafts.
    Forces revisions if quality threshold (score < 80) is not met.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Updated state dictionary with critic score and routing decisions
    """
    logger.info("NODE: critic")
    section_drafts = state.get("section_drafts", {})
    metadata = state.get("metadata", {})
    retrieved_documents = metadata.get("retrieved_documents", {})

    if not section_drafts or not retrieved_documents:
        logger.warning("No drafts or documents to review. Approving by default.")
        return {"critic_feedback": "Approved", "routing_decision": "generate_answer"}

    # Combine drafts and documents into a clean readable prompt
    full_draft = "\n\n".join(section_drafts.values())

    contexts = []
    for title, hits in retrieved_documents.items():
        contexts.append(f"Section: {title}")
        for i, hit in enumerate(hits[:2]):
            contexts.append(f"Source {i + 1} Text:\n{hit.get('parent_text') or hit.get('chunk_text') or ''}")
    context_str = "\n\n".join(contexts)

    prompt = (
        "You are an academic Referee/Peer Reviewer for a top AI conference. "
        "Your task is to review the draft literature review sections below against the retrieved source texts "
        "and score the draft on scientific accuracy and citation integrity.\n\n"
        "--- START DRAFT SECTIONS ---\n"
        f"{full_draft}\n"
        "--- END DRAFT SECTIONS ---\n\n"
        "--- START SOURCE TEXTS ---\n"
        f"{context_str}\n"
        "--- END SOURCE TEXTS ---\n\n"
        "Instructions:\n"
        "Evaluate the draft based on:\n"
        "1. Factuality & Hallucinations: Are there claims in the draft that are NOT supported by the source texts?\n"
        "2. Citation Integrity: Are all claims accompanied by proper citations, and do they link to real sources?\n\n"
        "Output your peer review strictly as a JSON object matching this schema:\n"
        "{\n"
        '  "score": 85, // Integer score from 0 to 100\n'
        '  "reason": "Approved" // Detailed feedback if score < 80, otherwise \'Approved\'\n'
        "}\n"
        "Write only the raw JSON. Do not include markdown code block formatting (like ```json or ```)."
    )

    try:
        model = runtime.context.llm_client.get_langchain_model(runtime.context.model_name, temperature=0.0)
        response = await model.ainvoke(prompt)
        content = response.content.strip()

        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        review_data = json.loads(content)
        score = int(review_data.get("score", 100))
        reason = review_data.get("reason", "Approved")

        logger.info(f"Critic Score: {score}/100, Result: {reason[:100]}...")

        # Loop counter to prevent infinite rewrites
        rewrite_attempts = metadata.get("rewrite_attempts", 0)
        max_rewrites = 2

        if score >= 80 or rewrite_attempts >= max_rewrites:
            if rewrite_attempts >= max_rewrites:
                logger.warning(f"Max rewrite attempts ({max_rewrites}) reached. Forcing approval.")
            return {
                "critic_feedback": "Approved",
                "routing_decision": "generate_answer",
                "metadata": {**metadata, "critic_score": score},
            }
        else:
            logger.warning(f"Critic rejected draft with score {score}. Routing back for query rewrite and search.")
            return {
                "critic_feedback": reason,
                "routing_decision": "rewrite_query",
                "metadata": {**metadata, "critic_score": score, "rewrite_attempts": rewrite_attempts + 1},
            }

    except Exception as e:
        logger.error(f"Failed to run critic peer review: {e}. Defaulting to approval.")
        return {"critic_feedback": "Approved", "routing_decision": "generate_answer"}
