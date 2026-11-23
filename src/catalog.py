from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.logging_utils import get_logger
from src.models import Product

log = get_logger(__name__)


def _opt_path(value) -> Optional[Path]:
    if value is None or pd.isna(value) or str(value).strip() in ("", "None"):
        return None
    return Path(str(value))


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
                product_id=int(row["product_id"]),
                name=row["name"],
                description=row["description"],
                price=float(row["price"]),
                color=row["color"],
                category=row["category"],
                gender=row["gender"],
                image_path=_opt_path(row.get("image_path")),
                size_chart_path=_opt_path(row.get("size_chart_path")),
            )
            self.products.append(product)
            self._id_map[product.product_id] = product

        missing_images = [p.product_id for p in self.products if p.image_path and not p.image_path.exists()]
        missing_charts = [p.product_id for p in self.products if p.size_chart_path and not p.size_chart_path.exists()]
        if missing_images:
            log.warning("%d products reference missing image files (e.g. ids %s)", len(missing_images), missing_images[:5])
        if missing_charts:
            log.warning("%d products reference missing size charts (e.g. ids %s)", len(missing_charts), missing_charts[:5])

    def get_product(self, product_id: int) -> Product:
        if product_id in self._id_map:
            return self._id_map[product_id]
        raise ValueError(f"Product not found: {product_id}")

    get_products = get_product  # backwards-compatible alias

    def __len__(self):
        return len(self.products)

    def __repr__(self):
        return f"ProductCatalog({len(self.products)} products)"
