import sys
import os

sys.path.append(os.path.abspath("."))

from src.evaluation.conflict_resolver import ConflictResolver

resolver = ConflictResolver()

pass1_result = {"marks": 6.5}

pass2_result = {"marks": 5.0}

result = resolver.resolve(pass1_result, pass2_result)

print(result)
