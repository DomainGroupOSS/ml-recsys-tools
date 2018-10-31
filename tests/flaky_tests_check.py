import os
import unittest
from collections import Counter

root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

n_flakes = 0
flakes = Counter()
for i in range(50):
    # run tests
    suite = unittest.TestLoader().discover(root_dir)
    result = unittest.TextTestRunner().run(suite)

    fails = result.failures + result.errors
    if fails:
        flakes[fails[0][0]._testMethodName] += 1
        n_flakes += 1

    print(f'flakiness: {n_flakes / (i + 1)}')
    print(f'problematic tests: {flakes}')