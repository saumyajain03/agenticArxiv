import asyncio
import logging
from typing import Any, Dict, List, Union

import logfire
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState

logger = logging.getLogger(__name__)


@logfire.instrument("node:writer", extract_args=False)
async def ainvoke_writer_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[dict, list]]:
    """Section Writer node.

    Takes retrieved documents for each section and drafts technical content
    concurrently, citing source papers properly.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Updated state dictionary with drafted sections
    """
    logger.info("NODE: writer")
    research_plan = state.get("research_plan", [])
    metadata = state.get("metadata", {})
    retrieved_documents = metadata.get("retrieved_documents", {})

    if not research_plan or not retrieved_documents:
        logger.warning("No research plan or retrieved documents. Skipping writer node.")
        return {}

    async def draft_section(section: Dict[str, Any]) -> Dict[str, str]:
        """Draft a single literature review section using the retrieved document contexts."""
        title = section.get("title", "")
        description = section.get("description", "")
        hits = retrieved_documents.get(title, [])

        if not hits:
            logger.warning(f"No papers retrieved for section '{title}'. Writing default draft.")
            return {"title": title, "draft": f"### {title}\n\nNo source papers retrieved to compile this section."}

        # Select the top 1 paper to minimize token usage and avoid Groq 413 Rate Limits
        top_hits = hits[:1]
        contexts = []
        for i, hit in enumerate(top_hits):
            # Fetch the parent text (full section) or standard chunk
            doc_context = hit.get("parent_text") or hit.get("chunk_text") or ""
            # Aggressively truncate to ~1500 characters (~300 words) to fit TPM limits
            doc_context = doc_context[:1500]
            
            authors = hit.get("authors", "Unknown Authors")
            arxiv_id = hit.get("arxiv_id", "")
            title_paper = hit.get("title", "Unknown Title")

            contexts.append(
                f"[Source {i + 1}]:\nPaper: {title_paper} ({arxiv_id})\nAuthors: {authors}\nContent:\n{doc_context}\n---"
            )

        context_str = "\n\n".join(contexts)

        prompt = (
            "You are an expert AI scientist writing a section of a comprehensive literature review paper.\n\n"
            f"Section Title: {title}\n"
            f"Section Objective: {description}\n\n"
            "Below is the relevant content extracted from scientific research papers for your reference:\n"
            f"{context_str}\n\n"
            "Task:\n"
            "Write a highly technical, rigorous synthesis for this section (about 250-350 words).\n"
            "Requirements:\n"
            "1. Focus strictly on synthesizing the source text. Do not make up facts or add outside concepts.\n"
            "2. Incorporate precise citation anchors referencing the sources by their paper details (e.g., [Authors, Year] or [arXiv ID]).\n"
            "3. Use a formal, academic tone suitable for publication in a top conference (e.g., NeurIPS/ICML).\n"
            "4. Start directly with the section body content. Do not write section titles or markdown headers."
        )

        try:
            model = runtime.context.llm_client.get_langchain_model(runtime.context.model_name, temperature=0.2)
            response = await model.ainvoke(prompt)
            draft = response.content.strip()
            logger.info(f"✓ Section '{title}' successfully drafted ({len(draft)} chars)")
            return {"title": title, "draft": f"### {title}\n\n{draft}"}
        except Exception as e:
            logger.error(f"Failed to draft section '{title}': {e}")
            return {"title": title, "draft": f"### {title}\n\nFailed to synthesize content due to LLM error."}

    # Draft all sections concurrently
    tasks = [draft_section(sec) for sec in research_plan]
    draft_results = await asyncio.gather(*tasks)

    section_drafts = {}
    for res in draft_results:
        section_drafts[res["title"]] = res["draft"]

    return {"section_drafts": section_drafts}
