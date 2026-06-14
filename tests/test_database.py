import sys
import os

sys.path.append(os.path.abspath("."))

from src.database.crud import get_results

results = get_results()

print("\n===== DATABASE RESULTS =====\n")

for row in results:
    print(row)
