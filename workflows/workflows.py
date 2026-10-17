"""
workflows/workflows.py
Temporal Workflow for DabbaETA Dispatch Orchestration.
Chains: load_orders -> predict_all (TabPFN) -> flag_risky -> explain (Gemma) -> draft_messages (Gemma)
with exponential backoff and resume capabilities.
"""

from datetime import timedelta
from typing import Dict, Any
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from workflows.activities import (
        load_orders_activity,
        predict_all_activity,
        flag_risky_activity,
        explain_activity,
        draft_messages_activity,
    )

TASK_QUEUE_NAME = "dabbaeta-dispatch-queue"

@workflow.defn
class DispatchWorkflow:
    def __init__(self):
        self.current_step = "INITIALIZED"

    @workflow.run
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        threshold = input_data.get("threshold", 0.45)
        limit = input_data.get("limit", 25)

        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=60),
            maximum_attempts=5,
        )

        self.current_step = "LOADING_ORDERS"
        orders = await workflow.execute_activity(
            load_orders_activity,
            limit,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        self.current_step = "PREDICTING_TABPFN"
        predictions = await workflow.execute_activity(
            predict_all_activity,
            orders,
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=retry_policy,
        )

        self.current_step = "FLAGGING_RISKY"
        risky_orders = await workflow.execute_activity(
            flag_risky_activity,
            {"predictions": predictions, "threshold": threshold},
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        self.current_step = "EXPLAINING_GEMMA"
        explanation = await workflow.execute_activity(
            explain_activity,
            risky_orders,
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=retry_policy,
        )

        self.current_step = "DRAFTING_MESSAGES"
        drafts = await workflow.execute_activity(
            draft_messages_activity,
            risky_orders,
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=retry_policy,
        )

        self.current_step = "COMPLETED"
        return {
            "status": "COMPLETED",
            "total_orders": len(orders),
            "risky_orders_count": len(risky_orders),
            "explanation": explanation,
            "predictions": predictions,
            "risky_orders": risky_orders,
            "draft_messages": drafts,
        }
