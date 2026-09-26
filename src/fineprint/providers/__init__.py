"""The three model interfaces and the one implementation of each.

Nothing outside this package imports boto3 or the anthropic SDK. Retrieval, ingestion and
answering ask `factory.get_embedder()` and `factory.get_chat_model()` for an object that
satisfies the protocols in `base.py`, which is what lets a test swap in a fake and what will
let part 3 add a second provider without touching a caller.
"""
