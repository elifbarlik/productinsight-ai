from pathlib import Path
from typing import List, Dict
from src.models import Product
import pandas as pd

class ProductCatalog:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.products: List[Product] = []
        self._id_map: Dict[int, Product] = {}
        self.load_from_csv()

    def load_from_csv(self):
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            product = Product(
                product_id=int(row['product_id']),
                name=row['name'],
                description=row['description'],
                price=float(row['price']),
                color=row['color'],
                image_path=Path(row['image_path']),
                size_chart_path=Path(row['size_chart_path']) if pd.notna(row['size_chart_path']) else None
            )
            self.products.append(product)
            self._id_map[product.product_id] = product

    def get_products(self, product_id: int) -> Product:
        if product_id in self._id_map:
            return self._id_map[product_id]
        raise ValueError(f'Product not found: {product_id}')

    def __len__(self):
        return len(self.products)

    def __repr__(self):
        return f'ProductCatalog({len(self.products)} products)'
