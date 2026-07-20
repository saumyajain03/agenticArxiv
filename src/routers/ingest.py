import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dateutil import parser as date_parser
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.dependencies import (
    get_arxiv_client,
    get_db_session,
    get_embeddings_service,
    get_opensearch_client,
    get_pdf_parser,
    get_pinecone_client,
)
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import ArxivPaper, PaperCreate
from src.schemas.pdf_parser.models import ParsedPaper, PdfContent
from src.services.arxiv.client import ArxivClient
from src.services.indexing.hybrid_indexer import HybridIndexingService
from src.services.indexing.text_chunker import TextChunker
from src.services.pdf_parser.parser import PDFParserService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


class IngestRequest(BaseModel):
    arxiv_id: str = Field(..., description="arXiv ID of the research paper (e.g. '1706.03762')", min_length=5, max_length=20)


class IngestResponse(BaseModel):
    success: bool
    arxiv_id: str
    paper_id: Optional[str] = None
    chunks_indexed: int = 0
    error: Optional[str] = None


def parse_date(published_date: str) -> datetime:
    try:
        return date_parser.parse(published_date)
    except Exception:
        return datetime.now(timezone.utc)


def serialize_parsed_content(pdf_content: PdfContent) -> Dict[str, Any]:
    return {
        "pdf_processed": True,
        "raw_text": pdf_content.raw_text,
        "sections": [s.model_dump() for s in pdf_content.sections],
        "figures": [f.model_dump() for f in pdf_content.figures],
        "tables": [t.model_dump() for t in pdf_content.tables],
        "parser_used": pdf_content.parser_used.value,
        "parser_metadata": pdf_content.metadata,
    }


@router.post("/arxiv", response_model=IngestResponse)
async def ingest_arxiv_paper(
    request: IngestRequest,
    arxiv_client: ArxivClient = Depends(get_arxiv_client),
    pdf_parser: PDFParserService = Depends(get_pdf_parser),
    db_session: Session = Depends(get_db_session),
    embeddings_client=Depends(get_embeddings_service),
    opensearch_client=Depends(get_opensearch_client),
    pinecone_client=Depends(get_pinecone_client),
) -> IngestResponse:
    """Download, parse, database-store, and vector-index a single arXiv paper on-demand."""
    arxiv_id = request.arxiv_id.strip()
    logger.info(f"API request to ingest paper: {arxiv_id}")

    try:
        # 1. Fetch metadata from arXiv
        paper = await arxiv_client.fetch_paper_by_id(arxiv_id)
        if not paper:
            return IngestResponse(success=False, arxiv_id=arxiv_id, error="Failed to fetch paper metadata from arXiv.")

        # 2. Download and parse PDF
        # This will use the docling parser locally, or automatically fall back to pypdfium2 in the cloud!
        parsed_paper = None
        try:
            temp_pdf_path = await arxiv_client.download_pdf(paper)
            if temp_pdf_path:
                parsed_paper = await pdf_parser.parse_pdf(temp_pdf_path)
                # Cleanup temporary file
                if temp_pdf_path.exists():
                    temp_pdf_path.unlink()
        except Exception as e:
            logger.error(f"Error parsing PDF for {arxiv_id}: {e}")
            # We will still proceed to store the metadata if parsing fails, but vector search won't work

        # 3. Store to PostgreSQL
        paper_repo = PaperRepository(db_session)
        published_date = parse_date(paper.published_date)

        paper_data = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "categories": paper.categories,
            "published_date": published_date,
            "pdf_url": paper.pdf_url,
        }

        if parsed_paper:
            parsed_content = serialize_parsed_content(parsed_paper)
            paper_data.update(parsed_content)
        else:
            paper_data.update(
                {
                    "pdf_processed": False,
                    "parser_metadata": {"note": "PDF processing skipped or failed"},
                }
            )

        paper_create = PaperCreate(**paper_data)
        stored_paper = paper_repo.upsert(paper_create)
        db_session.commit()
        paper_id = str(stored_paper.id)

        # 4. If we have parsed text, build indexing service and index to Pinecone/OpenSearch
        chunks_indexed = 0
        if parsed_paper:
            chunker = TextChunker()
            indexing_service = HybridIndexingService(
                chunker=chunker,
                embeddings_client=embeddings_client,
                opensearch_client=opensearch_client,
                pinecone_client=pinecone_client,
            )

            raw_text = parsed_paper.raw_text
            sections = [{"title": s.title, "content": s.content} for s in parsed_paper.sections]

            # We format index_paper parameters to look like the db payload
            indexing_payload = {
                "id": paper_id,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "categories": paper.categories,
                "raw_text": raw_text,
                "sections": sections,
                "published_date": published_date,
            }

            stats = await indexing_service.index_paper(indexing_payload)
            chunks_indexed = stats.get("chunks_indexed", 0)

        return IngestResponse(success=True, arxiv_id=arxiv_id, paper_id=paper_id, chunks_indexed=chunks_indexed)

    except Exception as e:
        logger.error(f"Unexpected error ingesting paper {arxiv_id}: {e}")
        db_session.rollback()
        return IngestResponse(success=False, arxiv_id=arxiv_id, error=str(e))
