from .base import EngineLM, CachedEngine

__ENGINE_NAME_SHORTCUTS__ = {
    "opus": "claude-3-opus-20240229",
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229",
    "together-llama-3-70b": "together-meta-llama/Llama-3-70b-chat-hf",
}

def get_engine(engine_name: str, **kwargs) -> EngineLM:
    if engine_name in __ENGINE_NAME_SHORTCUTS__:
        engine_name = __ENGINE_NAME_SHORTCUTS__[engine_name]

    if "seed" in kwargs and engine_name not in ["gpt-4", "gpt-3.5"]:
        raise ValueError(f"Seed is currently supported only for OpenAI engines, not {engine_name}")

    if (("gpt-4" in engine_name) or ("gpt-3.5" in engine_name)):
        from .openai import ChatOpenAI
        return ChatOpenAI(model_string=engine_name, **kwargs)    
    else:
        raise ValueError(f"Engine {engine_name} not supported")