from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Product:
    """E-commerce product model."""

    product_id: int
    name: str
    description: str
    price: float
    color: str
    category: str
    gender: str
    image_path: Optional[Path] = None
    size_chart_path: Optional[Path] = None

    def __post_init__(self):
        if self.price <= 0:
            raise ValueError("Price must be positive value")

    def embedding_text(self) -> str:
        """Text used for both dense embedding and BM25 indexing."""
        return (
            f"{self.name}. Category: {self.category}. Color: {self.color}. "
            f"Gender: {self.gender}. {self.description}"
        )

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "description": self.description,
            "price": self.price,
            "color": self.color,
            "category": self.category,
            "gender": self.gender,
            "image_path": str(self.image_path) if self.image_path else None,
            "size_chart_path": str(self.size_chart_path) if self.size_chart_path else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Product":
        return cls(
            product_id=int(d["product_id"]),
            name=d["name"],
            description=d["description"],
            price=float(d["price"]),
            color=d["color"],
            category=d.get("category", ""),
            gender=d.get("gender", ""),
            image_path=Path(d["image_path"]) if d.get("image_path") else None,
            size_chart_path=Path(d["size_chart_path"]) if d.get("size_chart_path") else None,
        )
