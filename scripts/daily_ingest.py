import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dateutil import parser as date_parser
from src.config import get_settings
from src.db.factory import make_database
from src.repositories.paper import PaperRepository
from src.services.arxiv.client import ArxivClient
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient
from src.services.embeddings.pinecone_client import PineconeClient
from src.services.indexing.hybrid_indexer import HybridIndexingService
from src.services.indexing.text_chunker import TextChunker
from src.schemas.arxiv.paper import PaperCreate

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def serialize_parsed_content(parsed_paper):
    return {
        "pdf_processed": True,
        "raw_text": parsed_paper.raw_text,
        "sections": [s.model_dump() for s in parsed_paper.sections],
        "figures": [f.model_dump() for f in parsed_paper.figures],
        "tables": [t.model_dump() for t in parsed_paper.tables],
        "parser_used": parsed_paper.parser_used.value,
        "parser_metadata": parsed_paper.metadata,
    }


async def run_ingestion():
    logger.info("Starting scheduled daily paper ingestion...")

    settings = get_settings()

    # 1. Initialize DB and Repository
    database = make_database()
    db_session = database.session_factory()
    paper_repo = PaperRepository(db_session)

    # 2. Check document count to decide daily check vs historical catch-up
    try:
        paper_count = paper_repo.get_count()
        logger.info(f"Current database contains {paper_count} papers")
    except Exception as e:
        logger.error(f"Error checking database paper count: {e}")
        db_session.close()
        return

    # Dynamic date range calculation
    now = datetime.now(timezone.utc)
    if paper_count == 0:
        logger.info("Database is empty. Running a 14-day historical catch-up load...")
        start_date = now - timedelta(days=14)
        max_results = 30
    else:
        logger.info("Database is populated. Running a 2-day daily rolling check...")
        start_date = now - timedelta(days=2)
        max_results = 20

    # Build search query targeting the 5 core CS categories
    search_query = settings.arxiv.search_category or "cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:cs.NE"
    logger.info(f"Target categories: {search_query}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")

    # 3. Fetch list of papers from arXiv
    arxiv_client = ArxivClient(settings)
    papers = await arxiv_client.fetch_papers(
        max_results=max_results,
        from_date=start_date.strftime("%Y%m%d"),
        to_date=now.strftime("%Y%m%d"),
    )

    if not papers:
        logger.info("No new papers found in the date range. Ingestion complete.")
        db_session.close()
        return

    logger.info(f"Found {len(papers)} candidate papers from arXiv. Starting ingestion...")

    # 4. Initialize Indexing Service
    embeddings_client = JinaEmbeddingsClient(api_key=settings.jina_api_key)
    
    opensearch_client = None
    pinecone_client = None

    if settings.vector_db_provider == "pinecone":
        pinecone_client = PineconeClient(
            api_key=settings.pinecone.api_key,
            index_name=settings.pinecone.index_name,
            environment=settings.pinecone.environment,
        )
    else:
        opensearch_client = OpenSearchClient(host=settings.opensearch.host, settings=settings)

    chunker = TextChunker()
    indexing_service = HybridIndexingService(
        chunker=chunker,
        embeddings_client=embeddings_client,
        opensearch_client=opensearch_client,
        pinecone_client=pinecone_client,
    )

    pdf_parser = make_pdf_parser_service()

    # 5. Process and index each paper
    success_count = 0
    for idx, paper in enumerate(papers):
        arxiv_id = paper.arxiv_id
        logger.info(f"[{idx+1}/{len(papers)}] Processing paper: {arxiv_id} - '{paper.title[:60]}...'")

        # Skip if already exists in postgres
        if paper_repo.get_by_arxiv_id(arxiv_id):
            logger.info(f"Paper {arxiv_id} already exists in database. Skipping.")
            continue

        parsed_paper = None
        temp_pdf_path = None
        try:
            temp_pdf_path = await arxiv_client.download_pdf(paper)
            if temp_pdf_path:
                parsed_paper = await pdf_parser.parse_pdf(temp_pdf_path)
        except Exception as e:
            logger.error(f"Failed to parse PDF for {arxiv_id}: {e}")
        finally:
            if temp_pdf_path and temp_pdf_path.exists():
                temp_pdf_path.unlink()

        try:
            # Store to postgres
            paper_data = {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "categories": paper.categories,
                "published_date": parse_date(paper.published_date),
                "pdf_url": paper.pdf_url,
            }

            if parsed_paper:
                paper_data.update(serialize_parsed_content(parsed_paper))
            else:
                paper_data.update({
                    "pdf_processed": False,
                    "parser_metadata": {"note": "PDF processing failed during scheduled check"},
                })

            paper_create = PaperCreate(**paper_data)
            stored_paper = paper_repo.upsert(paper_create)
            db_session.commit()
            paper_id = str(stored_paper.id)

            # Vector Indexing
            if parsed_paper:
                indexing_payload = {
                    "id": paper_id,
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "raw_text": parsed_paper.raw_text,
                    "sections": [{"title": s.title, "content": s.content} for s in parsed_paper.sections],
                    "published_date": parse_date(paper.published_date),
                }

                stats = await indexing_service.index_paper(indexing_payload)
                logger.info(f"Successfully vector-indexed paper {arxiv_id} with {stats['chunks_indexed']} chunks")
            
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to store/index paper {arxiv_id}: {e}")
            db_session.rollback()

    logger.info(f"Ingestion completed. Successfully processed {success_count}/{len(papers)} new papers.")
    db_session.close()


def parse_date(published_date: str) -> datetime:
    try:
        return date_parser.parse(published_date)
    except Exception:
        return datetime.now(timezone.utc)


if __name__ == "__main__":
    asyncio.run(run_ingestion())
