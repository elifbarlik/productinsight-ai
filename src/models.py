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
    image_path: Path
    size_chart_path: Optional[Path]=None
    def __post_init__(self):
        """Validation"""
        assert self.price >0 , "Price must be positive value"
        assert self.image_path.exists(), f"Image path does not exist: {self.image_path}"
        if self.size_chart_path:
            assert self.size_chart_path.exists(), f"Size chart path does not exist: {self.size_chart_path}"
    def to_dict(self):
        return{
            'id':self.product_id,
            'name':self.name,
            'description':self.description,
            'price':self.price,
            'color':self.color,
            'image_path':self.image_path,
            'size_chart_path':str(self.size_chart_path) if self.size_chart_path else None
        }