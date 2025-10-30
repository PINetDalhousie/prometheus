
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

#file = open('MS_thesis-main/logs/20240409_054150/training.log', 'r')

f1_scores = [0.897, 0.0192, 0.00174, 0.00174, 0.00174, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',]

plt.figure(figsize=(6, 4))
plt.plot(labels, f1_scores, marker='o', linestyle='-', color='b')

# Add labels to each point
points=['','uva', 'bbe', 'rxmax', 'cap']
c=0
for x, y in zip(labels, f1_scores):
    if c<4:
        plt.text(x, y, f'{points[c]}', fontsize=10, ha='center', va='bottom')
    else:
        plt.text(x, y,f'{ ""}', fontsize=10, ha='center', va='bottom')
    c+=1

plt.title('F1 Score Comparison', fontsize=16)
plt.xlabel('# Removed features', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('f1_score_comparison.pdf')
plt.show()