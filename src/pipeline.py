import subprocess, sys
steps = [
    ['python','src/data_prep.py'],
    ['python','src/train.py'],
    ['python','src/fairness_metrics.py'],
    ['python','src/mitigate.py'],
]
for cmd in steps:
    print('>>', ' '.join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        sys.exit(res.returncode)
print('Pipeline finished.')
