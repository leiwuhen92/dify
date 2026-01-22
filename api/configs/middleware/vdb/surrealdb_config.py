from typing import Optional

from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings


class SurrealDBConfig(BaseSettings):
    """
    Configuration settings for SurrealDB vector database
    """

    SURREALDB_URL: Optional[str] = Field(
        description="URL of the SurrealDB server (e.g., 'http://localhost:8000' or ws://localhost:8000/rpc')",
        default=None,
    )

    SURREALDB_USERNAME: Optional[str] = Field(
        description="Username for authenticating with the SurrealDB server",
        default=None,
    )

    SURREALDB_PASSWORD: Optional[str] = Field(
        description="Password for authenticating with the SurrealDB server",
        default=None,
    )

    SURREALDB_NAMESPACE: Optional[str] = Field(
        description="Namespace of the SurrealDB server",
        default="cloud_ns",
    )

    SURREALDB_DATABASE: Optional[str] = Field(
        description="Name of the SurrealDB database to connect to(default is 'default')",
        default="cloud_db",
    )

    SURREALDB_BATCH_SIZE: PositiveInt = Field(
        description="Number of objects to be processed in a single batch operation (default is 100)",
        default=4,
    )

    SURREALDB_ENABLE_HYBRID_SEARCH: bool = Field(
        description="Enable hybrid search features. Set to false for compatibility with older versions",
        default=True,
    )
