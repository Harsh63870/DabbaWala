"""
workflows/worker.py
Temporal Worker for DabbaETA.
Listens on 'dabbaeta-dispatch-queue' and executes DispatchWorkflow activities.
Instrumented with Sentry tracing.
"""

import os
import sys
import asyncio
import logging
import sentry_sdk
from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

load_dotenv()

from workflows.workflows import DispatchWorkflow, TASK_QUEUE_NAME
from workflows.activities import (
    load_orders_activity,
    predict_all_activity,
    flag_risky_activity,
    explain_activity,
    draft_messages_activity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dabbaeta.worker")

SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=1.0,
        release="dabbaeta@1.0.0",
        environment=os.getenv("ENVIRONMENT", "development")
    )
    logger.info("Sentry initialized for Temporal Worker.")

async def run_worker():
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    logger.info(f"Connecting Temporal Worker to {temporal_host}...")

    try:
        client = await Client.connect(temporal_host)
        logger.info(f"Connected to Temporal server. Registering worker on queue '{TASK_QUEUE_NAME}'...")

        worker = Worker(
            client,
            task_queue=TASK_QUEUE_NAME,
            workflows=[DispatchWorkflow],
            activities=[
                load_orders_activity,
                predict_all_activity,
                flag_risky_activity,
                explain_activity,
                draft_messages_activity,
            ],
        )

        logger.info("🚀 Temporal Worker is listening for dispatch jobs. Press Ctrl+C to stop.")
        await worker.run()

    except Exception as e:
        logger.error(f"Failed to start worker or connect to Temporal server at {temporal_host}: {e}")
        logger.info("Tip: Start local dev server via `temporal server start-dev`.")
        raise e

if __name__ == "__main__":
    asyncio.run(run_worker())
