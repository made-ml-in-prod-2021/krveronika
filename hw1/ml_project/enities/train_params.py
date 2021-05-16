from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=42)
        