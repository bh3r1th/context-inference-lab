import asyncio
import json

import httpx

from contextbench.experiments.config import Server
from contextbench.context import make_strategy
from contextbench.inference.vllm import VLLMBackend
from contextbench.models import Context


def call_response(body, status=200):
    async def execute():
        def handler(request):
            payload = json.loads(request.content)
            assert payload["stream_options"]["include_usage"]
            assert "json" in payload["structured_outputs"]
            return httpx.Response(status, text=body)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            config = Server(model_revision="abc", vllm_version="test", launch_command="test")
            return await VLLMBackend(config, client).generate(Context("rules", ()), {}, 42)
    return asyncio.run(execute())


def test_sse_usage_and_role_only_chunk():
    result = call_response('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                           'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\n'
                           'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}\n\n'
                           'data: [DONE]\n\n')
    assert result.error is None
    assert result.text == "{}" and result.input_tokens == 12
    assert result.ttft_s is not None
    assert result.decode_tokens_per_s is None  # one content chunk cannot measure decode
    assert result.prefill_s is None


def test_truncated_stream():
    assert "incomplete" in call_response('data: {"choices":[]}\n\n').error


def test_http_error():
    assert "HTTPStatusError" in call_response("unavailable", 503).error


def test_missing_usage_is_explicit():
    result = call_response('data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\n'
                           'data: [DONE]\n\n')
    assert result.input_tokens is None and "input_tokens" in result.unsupported


def test_backend_uses_inspected_messages_without_duplicate_event(dataset):
    event = {"event_id": "ONE-EVENT", "city": "Austin"}
    context = make_strategy("full_context", dataset).build(event)

    async def execute():
        def handler(request):
            payload = json.loads(request.content)
            assert payload["messages"] == list(context.messages)
            assert sum(m["content"].count("ONE-EVENT") for m in payload["messages"]) == 1
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"{}"},'
                                  '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            config = Server(model_revision="abc", vllm_version="test", launch_command="test")
            return await VLLMBackend(config, client).generate(context, event, 42)

    assert asyncio.run(execute()).error is None
