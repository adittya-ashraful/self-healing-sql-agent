"""Scripted fake ChatModel supporting `.with_structured_output(SQLGeneration)` for deterministic tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from app.schemas import SQLGeneration


def _flatten_messages(messages: Any) -> str:
    """Concatenate a list of LangChain messages into a single string for inspection."""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        return "\n".join(str(getattr(m, "content", m)) for m in messages)
    return str(messages)


class ScriptedSQLChatModel(BaseChatModel):
    """Deterministic :class:`BaseChatModel` for testing SQL generation, repair, and summarization.

    Each call to the structured-output runnable pops the next :class:`SQLGeneration`
    from ``sql_steps`` (repeating the last one if exhausted) and records the prompt text
    in ``structured_prompts``. Direct ``invoke`` / ``ainvoke`` calls return ``summary_text``
    (or call ``summary_fn``).
    """

    sql_steps: list[SQLGeneration] = Field(default_factory=list)
    summary_text: str = "Last month's revenue was $834.70."
    summary_fn: Callable[[str], str] | None = None
    structured_prompts: list[str] = Field(default_factory=list)
    summary_prompts: list[str] = Field(default_factory=list)
    _cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-sql-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        prompt_text = _flatten_messages(messages)
        self.summary_prompts.append(prompt_text)
        reply = self.summary_fn(prompt_text) if self.summary_fn is not None else self.summary_text
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        del schema, include_raw, kwargs

        async def _ainvoke(inputs: Any) -> SQLGeneration:
            self.structured_prompts.append(_flatten_messages(inputs))
            if not self.sql_steps:
                raise RuntimeError("ScriptedSQLChatModel has no configured sql_steps")
            idx = min(self._cursor, len(self.sql_steps) - 1)
            self._cursor += 1
            return self.sql_steps[idx]

        def _invoke(inputs: Any) -> SQLGeneration:
            self.structured_prompts.append(_flatten_messages(inputs))
            if not self.sql_steps:
                raise RuntimeError("ScriptedSQLChatModel has no configured sql_steps")
            idx = min(self._cursor, len(self.sql_steps) - 1)
            self._cursor += 1
            return self.sql_steps[idx]

        return RunnableLambda(func=_invoke, afunc=_ainvoke)
