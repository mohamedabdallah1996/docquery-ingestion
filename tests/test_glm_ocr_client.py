import httpx

from docquery_ingestion.clients.glm_ocr import LlamaCppOCRClient


def _client(handler, *, retry_max_attempts: int = 3) -> LlamaCppOCRClient:
    return LlamaCppOCRClient(
        service_url="http://fake",
        model="m",
        prompt="OCR",
        temperature=0.1,
        top_k=1,
        request_timeout_s=5,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_seconds=0.01,  # fast retries in tests
        transport=httpx.MockTransport(handler),
    )


async def test_successful_call_returns_cleaned_markdown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "```markdown\n# Hi\n```"}}]}
        )

    client = _client(handler)
    result = await client.extract_page(b"image", page_number=1)

    assert result.error is None
    assert result.markdown == "# Hi"
    assert result.page_number == 1


async def test_transient_failure_recovers_on_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = _client(handler, retry_max_attempts=5)
    result = await client.extract_page(b"image", page_number=1)

    assert calls["n"] == 3
    assert result.error is None
    assert result.markdown == "ok"


async def test_exhausted_retries_degrade_gracefully_not_an_exception() -> None:
    def always_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = _client(always_503, retry_max_attempts=2)
    result = await client.extract_page(b"image", page_number=7)  # must not raise

    assert result.page_number == 7
    assert result.markdown == ""
    assert result.error is not None


async def test_malformed_response_degrades_gracefully_without_retrying() -> None:
    calls = {"n": 0}

    def malformed(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _client(malformed, retry_max_attempts=5)
    result = await client.extract_page(b"image", page_number=1)

    assert calls["n"] == 1  # not retried -- a malformed body won't fix itself
    assert result.error is not None


async def test_sends_the_documented_request_shape() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = _client(handler)
    await client.extract_page(b"fake-image-bytes", page_number=1)

    body = captured["body"]
    assert isinstance(body, dict)
    message = body["messages"][0]
    assert message["role"] == "user"
    types = {part["type"] for part in message["content"]}
    assert types == {"image_url", "text"}
