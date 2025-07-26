from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    checkpoint: str
    drain_state: str
    vocab_size: int
    topk: int = 5
    r: int = 3
    mask_ratio: float = 0.3

    class Config:
        env_file = ".env"

settings = AppSettings()
