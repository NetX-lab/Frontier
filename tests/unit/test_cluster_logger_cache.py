import logging
import uuid

from frontier import logger as frontier_logger


def test_get_cluster_logger_reuses_adapter_for_same_context() -> None:
    logger_name = f"frontier.tests.cluster_logger.{uuid.uuid4()}"

    first = frontier_logger.get_cluster_logger(logger_name, "PREFILL")
    second = frontier_logger.get_cluster_logger(logger_name, "PREFILL")

    assert first is second


def test_get_cluster_logger_keeps_contexts_isolated() -> None:
    logger_name = f"frontier.tests.cluster_logger.{uuid.uuid4()}"

    prefill = frontier_logger.get_cluster_logger(logger_name, "PREFILL")
    decode = frontier_logger.get_cluster_logger(logger_name, "DECODE_ATTN")
    unscoped = frontier_logger.get_cluster_logger(logger_name)

    assert prefill is not decode
    assert isinstance(prefill, logging.LoggerAdapter)
    assert isinstance(decode, logging.LoggerAdapter)
    assert unscoped is logging.getLogger(logger_name)


def test_cluster_logger_adapter_preserves_existing_process_semantics() -> None:
    logger_name = f"frontier.tests.cluster_logger.{uuid.uuid4()}"
    adapter = frontier_logger.get_cluster_logger(logger_name, "PREFILL")

    message, kwargs = adapter.process(
        "scheduled",
        {"extra": {"request_id": 7, "cluster_type": "DECODE_ATTN"}},
    )

    assert message == "scheduled"
    assert kwargs["extra"] == {"request_id": 7, "cluster_type": "PREFILL"}


def test_cluster_logger_adapter_adds_missing_extra_context() -> None:
    logger_name = f"frontier.tests.cluster_logger.{uuid.uuid4()}"
    adapter = frontier_logger.get_cluster_logger(logger_name, "DECODE_FFN")

    message, kwargs = adapter.process("scheduled", {})

    assert message == "scheduled"
    assert kwargs["extra"] == {"cluster_type": "DECODE_FFN"}
