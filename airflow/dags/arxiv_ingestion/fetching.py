import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from .common import get_cached_services

logger = logging.getLogger(__name__)

async def run_paper_ingestion_pipeline(
    target_date: str,
    process_pdfs: bool = True,
) -> dict:
    """Run the paper ingestion pipeline for the target date.

    :param target_date: Target execution date (YYYYMMDD format)
    :param process_pdfs: Whether to download and process PDFs
    :returns: Dictionary with ingestion statistics
    """
    arxiv_client, _, database, metadata_fetcher, _ = get_cached_services()

    # Parse target date
    try:
        target_dt = datetime.strptime(target_date, "%Y%m%d")
    except ValueError:
        target_dt = datetime.now()
        target_date = target_dt.strftime("%Y%m%d")

    # Check database to see if we have existing papers.
    # If database is empty, do a 14-day historical load to index papers "till date".
    # Otherwise, check the last 2 days of submissions (covering weekend/timezone gaps).
    with database.get_session() as session:
        from src.models.paper import Paper
        paper_count = session.query(Paper).count()

        if paper_count == 0:
            logger.info("Database is empty. Initiating 14-day historical catch-up load...")
            from_dt = target_dt - timedelta(days=14)
            max_res = 30
        else:
            logger.info("Database is populated. Performing 2-day daily check...")
            from_dt = target_dt - timedelta(days=2)
            max_res = 20

        from_date = from_dt.strftime("%Y%m%d")
        to_date = target_date

        logger.info(f"Ingesting papers submitted from {from_date} to {to_date} (max_results={max_res})")

        return await metadata_fetcher.fetch_and_process_papers(
            max_results=max_res,
            from_date=from_date,
            to_date=to_date,
            process_pdfs=process_pdfs,
            store_to_db=True,
            db_session=session,
        )


def fetch_daily_papers(**context):
    """Fetch daily papers from arXiv and store in PostgreSQL.

    This task:
    1. Fetches papers from arXiv API using a hardcoded keyword search
    2. Downloads and processes PDFs using Docling
    3. Stores metadata and parsed content in PostgreSQL

    Note: OpenSearch indexing is handled by a separate dedicated task
    """
    logger.info("Starting daily paper fetching task (keyword search mode)")

    target_date = datetime.now().strftime("%Y%m%d")
    logger.info(f"Run date (for reference): {target_date}")

    results = asyncio.run(
        run_paper_ingestion_pipeline(
            target_date=target_date,
            process_pdfs=True,
        )
    )

    logger.info(f"Daily fetch complete: {results['papers_fetched']} papers fetched")

    results["date"] = target_date
    ti = context.get("ti")
    if ti:
        ti.xcom_push(key="fetch_results", value=results)

    return results
