import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# Ensure src/ is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dateutil import parser as date_parser
from sqlalchemy import delete, select
from src.config import get_settings
from src.db.factory import make_database
from src.models.paper import Paper
from src.repositories.paper import PaperRepository
from src.services.arxiv.client import ArxivClient
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.embeddings.pinecone_client import PineconeClient
from src.services.indexing.hybrid_indexer import HybridIndexingService
from src.services.indexing.text_chunker import TextChunker
from src.services.opensearch.client import OpenSearchClient
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.schemas.arxiv.paper import PaperCreate

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration from environment ────────────────────────────────────────────
# Which arXiv categories to ingest.
# Override via ARXIV_CATEGORIES env var in Render dashboard without a redeploy.
ARXIV_CATEGORIES = os.environ.get(
    "ARXIV_CATEGORIES",
    "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:cs.NE",
)

# Papers older than this many days are pruned from Pinecone + PostgreSQL.
# Default: 90 days (~54k vectors at 20 papers/day — safely within Pinecone free tier).
INGEST_TTL_DAYS = int(os.environ.get("INGEST_TTL_DAYS", "90"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_date(published_date: str) -> datetime:
    try:
        return date_parser.parse(published_date)
    except Exception:
        return datetime.now(timezone.utc)


def serialize_parsed_content(parsed_paper) -> dict:
    return {
        "pdf_processed": True,
        "raw_text": parsed_paper.raw_text,
        "sections": [s.model_dump() for s in parsed_paper.sections],
        "figures": [f.model_dump() for f in parsed_paper.figures],
        "tables": [t.model_dump() for t in parsed_paper.tables],
        "parser_used": parsed_paper.parser_used.value,
        "parser_metadata": parsed_paper.metadata,
    }


# ── Cleanup ───────────────────────────────────────────────────────────────────

async def cleanup_old_data(
    db_session,
    pinecone_client: Optional[PineconeClient],
    ttl_days: int,
) -> None:
    """
    Delete papers (and their Pinecone vectors) older than ``ttl_days`` days.

    Strategy:
    1. Fetch old paper IDs + arxiv_ids from PostgreSQL.
    2. Delete each paper's chunk vectors from Pinecone (by constructed vector ID).
    3. Bulk-delete the rows from PostgreSQL.

    This keeps both databases in sync and stays within free-tier limits.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    logger.info(f"[Cleanup] Pruning papers published before {cutoff.strftime('%Y-%m-%d')} (TTL={ttl_days} days)...")

    # 1. Identify old papers
    stmt = select(Paper.id, Paper.arxiv_id, Paper.title).where(Paper.published_date < cutoff)
    old_papers = db_session.execute(stmt).all()

    if not old_papers:
        logger.info("[Cleanup] No old papers to prune. Index is within TTL window.")
        return

    logger.info(f"[Cleanup] Found {len(old_papers)} papers to prune.")

    # 2. Delete their vectors from Pinecone
    if pinecone_client:
        for row in old_papers:
            paper_id = str(row.id)
            # Chunk IDs follow the pattern set by HybridIndexingService:
            # "{paper_id}_chunk_{i}" — we probe up to 200 chunks per paper
            # (real papers rarely exceed 100 chunks)
            vector_ids = [f"{paper_id}_chunk_{i}" for i in range(200)]
            try:
                pinecone_client.index.delete(ids=vector_ids, namespace="")
                logger.info(f"[Cleanup] Deleted Pinecone vectors for paper {row.arxiv_id}")
            except Exception as e:
                logger.warning(f"[Cleanup] Could not delete Pinecone vectors for {row.arxiv_id}: {e}")

    # 3. Bulk-delete from PostgreSQL
    paper_ids = [row.id for row in old_papers]
    db_session.execute(delete(Paper).where(Paper.id.in_(paper_ids)))
    db_session.commit()
    logger.info(f"[Cleanup] Deleted {len(old_papers)} papers from PostgreSQL.")


# ── Main ingestion pipeline ───────────────────────────────────────────────────

async def run_ingestion() -> None:
    logger.info("=" * 60)
    logger.info("  Daily arXiv Ingestion — Render Cron Job")
    logger.info(f"  Categories : {ARXIV_CATEGORIES}")
    logger.info(f"  TTL        : {INGEST_TTL_DAYS} days")
    logger.info("=" * 60)

    settings = get_settings()

    # 1. Initialize DB + Repository
    database = make_database()
    db_session = database.session_factory()
    paper_repo = PaperRepository(db_session)

    # 2. Check document count → decide fetch window
    try:
        paper_count = paper_repo.get_count()
        logger.info(f"Current database contains {paper_count} papers")
    except Exception as e:
        logger.error(f"Error checking database paper count: {e}")
        db_session.close()
        return

    now = datetime.now(timezone.utc)
    if paper_count == 0:
        logger.info("Database is empty. Running a 14-day historical catch-up...")
        start_date = now - timedelta(days=14)
        max_results = 30
    else:
        logger.info("Database is populated. Running 2-day rolling daily check...")
        start_date = now - timedelta(days=2)
        max_results = 20

    logger.info(f"Date range : {start_date.strftime('%Y-%m-%d')} → {now.strftime('%Y-%m-%d')}")

    # 3. Initialize services
    embeddings_client = JinaEmbeddingsClient(api_key=settings.jina_api_key)

    pinecone_client: Optional[PineconeClient] = None
    opensearch_client = None

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

    # 4. Fetch papers from arXiv
    arxiv_client = ArxivClient(settings)
    logger.info(f"Fetching up to {max_results} papers from arXiv...")
    papers = await arxiv_client.fetch_papers(
        max_results=max_results,
        from_date=start_date.strftime("%Y%m%d"),
        to_date=now.strftime("%Y%m%d"),
    )

    if not papers:
        logger.info("No new papers found in the date range.")
    else:
        logger.info(f"Found {len(papers)} candidate papers. Starting ingestion...")

        # 5. Process and index each paper
        success_count = 0
        for idx, paper in enumerate(papers):
            arxiv_id = paper.arxiv_id
            logger.info(f"[{idx + 1}/{len(papers)}] {arxiv_id} — {paper.title[:60]}...")

            # Skip duplicates
            if paper_repo.get_by_arxiv_id(arxiv_id):
                logger.info(f"  ↳ Already indexed. Skipping.")
                continue

            # Download + parse PDF
            parsed_paper = None
            temp_pdf_path = None
            try:
                temp_pdf_path = await arxiv_client.download_pdf(paper)
                if temp_pdf_path:
                    parsed_paper = await pdf_parser.parse_pdf(temp_pdf_path)
            except Exception as e:
                logger.error(f"  ↳ PDF parse failed: {e}")
            finally:
                if temp_pdf_path and temp_pdf_path.exists():
                    temp_pdf_path.unlink()

            # Persist to PostgreSQL
            try:
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
                        "parser_metadata": {"note": "PDF processing failed during scheduled ingestion"},
                    })

                paper_create = PaperCreate(**paper_data)
                stored_paper = paper_repo.upsert(paper_create)
                db_session.commit()
                paper_id = str(stored_paper.id)

                # Embed + upsert to Pinecone
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
                    logger.info(f"  ↳ Indexed {stats['chunks_indexed']} chunks into Pinecone.")

                success_count += 1

            except Exception as e:
                logger.error(f"  ↳ Failed to store/index {arxiv_id}: {e}")
                db_session.rollback()

        logger.info(f"Ingestion complete: {success_count}/{len(papers)} new papers processed.")

    # 6. Cleanup phase — runs every time regardless of new paper count
    logger.info("-" * 60)
    await cleanup_old_data(db_session, pinecone_client, ttl_days=INGEST_TTL_DAYS)
    logger.info("-" * 60)

    db_session.close()
    logger.info("All done. Cron job finished successfully.")


if __name__ == "__main__":
    asyncio.run(run_ingestion())
