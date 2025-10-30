import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example dataset of F1-scores for three different models
data = {
    "Model": ["Gentrap"] * 5 + ["LTrans"] * 5 + ["LSTM+"] * 5 +  ["LLSTM+"] * 5,
    "F1-score": [
        0.713, 0.682, 0.637, 0.667, 0.695,   # Model A
        0.804, 0.792, 0.867, 0.814, 0.846,   # Model B
        0.418, 0.325, 0.204, 0.134, 0.165, 
        0.408, 0.306, 0.210, 0.288, 0.250
    ]
}


data2 = {
    "Model": ["Gentrap"] * 5 + ["LTrans"] * 5 + ["LSTM+"] * 5 +  ["LLSTM+"] * 5,
    "F1-score": [
        0.906, 0.887, 0.936, 0.874, 0.886,   # Model A
        0.950, 0.936, 0.951, 0.913, 0.933,   # Model B
        0.512, 0.543, 0.611, 0.218, 0.556, 
        0.610, 0.680, 0.782, 0.305, 0.572
    ]
}
df1 = pd.DataFrame(data)
df1['Source'] = 'Urban'
df2 = pd.DataFrame(data2)
df2['Source'] = 'Rural'

# Combine the two DataFrames
df_combined = pd.concat([df2, df1], ignore_index=True)

# Create the boxplot with hue
plt.figure(figsize=(12, 7))
#sns.boxplot(x="Model", y="F1-score", hue="Source", data=df_combined, palette="Set2")
colors = sns.color_palette("colorblind", 4)
palette = dict(zip(
    ["Gentrap", "LTrans", "LSTM+", "LLSTM+"],
    colors
))

ax = sns.boxplot(
    x="Source",
    y="F1-score",
    hue="Model",
    data=df_combined,
    palette=palette,
    dodge=True
)

# Make boxplot borders bold
for patch in ax.artists:
    patch.set_linewidth(3.5)
    patch.set_edgecolor('black')
#plt.title("Distribution and Variability of F1-scores Across Models", fontsize=18, fontweight='bold')
legend = ax.legend_
if legend is not None:
    for text in legend.get_texts():
        text.set_fontsize(22)
       # text.set_fontweight('bold')
    legend.set_title(legend.get_title().get_text(), prop={'size': 22, 'weight': 'bold'})

plt.xlabel("Model", fontsize=24, fontweight='bold', labelpad=20)      # Increased labelpad for space
plt.ylabel("F1-score", fontsize=24, fontweight='bold', labelpad=20)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('boxplot_combined_final_invert.pdf')