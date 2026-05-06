"""Backends for the Razer customer-companion benchmark.

Each backend exposes ``ingest_sessions(...)`` to load the conversation arc
into whatever memory representation it uses, and ``ask(...)`` to answer a
single QA question as a specific user on a specific device. The harness
calls these uniformly so the benchmark itself stays backend-agnostic.
"""
