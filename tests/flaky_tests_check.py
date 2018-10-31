import os
import unittest


root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

flakes = 0
n_runs = 100
for i in range(n_runs):
    # run tests
    suite = unittest.TestLoader().discover(root_dir)
    result = unittest.TextTestRunner().run(suite)

    if result.failures:
        flakes += 1

print(f'flakiness: {flakes / n_runs}')