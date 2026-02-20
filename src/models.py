import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Product:
    """E-commerce product model"""
    product_id:int
    name:str
    description:str
    price:float
    color:str
    category:str
    gender:str
    image_path: Path
    size_chart_path: Optional[Path]=None
    def __post_init__(self):
        if self.price <= 0:
            raise ValueError("Price must be positive value")
        if self.image_path is not None and not self.image_path.exists():
            warnings.warn(f"Image path does not exist: {self.image_path}")
        if self.size_chart_path and not self.size_chart_path.exists():
            warnings.warn(f"Size chart path does not exist: {self.size_chart_path}")
    def to_dict(self):
        return{
            'product_id':self.product_id,
            'name':self.name,
            'description':self.description,
            'price':self.price,
            'color':self.color,
            'category':self.category,
            'gender':self.gender,
            'image_path':str(self.image_path),
            'size_chart_path':str(self.size_chart_path) if self.size_chart_path else None
        }