from pydantic import BaseModel

class CustomerData(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    hour: int
    day: int
    month: int
    year: int
