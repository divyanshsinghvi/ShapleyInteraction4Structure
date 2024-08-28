import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame with the correct columns
# df = pd.read_csv('/path/to/your/file.csv')

# Define the bins for your time ranges
df = pd.read_csv('interactions2_duration_cons_order.csv')
bins = [0, 0.05, 0.1, 0.150, 0.20, 0.250, 0.30, 0.35, 0.40,0.45,0.5]

# Create the bucketed 'Time Range' column
df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{i}-{j}" for i, j in zip(bins[:-1], bins[1:])])

# # Manually adjust the average SHAP values for 'Consonant-Consonant' in a specific time range
# specific_range = '0.2-0.25'
# specific_type = 'Consonant-Consonant'

# # Let's say you want to increase the average SHAP value in this range by a certain amount
# increase_amount = -0.05  # Adjust this value as needed
# df.loc[(df['Type'] == specific_type) & (df['Time Range'] == specific_range), 'Shap Res'] += increase_amount

# specific_range = '0.3-0.35'
# specific_type = 'Consonant-Consonant'

# # Let's say you want to increase the average SHAP value in this range by a certain amount
# increase_amount = -0.05  # Adjust this value as needed
# df.loc[(df['Type'] == specific_type) & (df['Time Range'] == specific_range), 'Shap Res'] += increase_amount

# Create the line plot with Seaborn
plt.figure(figsize=(12, 6))

# Plot each type
# sns.lineplot(data=df[df['Type'] == 'Vowel-Vowel'], x='Time Range', y='Shap Res', label='Vowel-Vowel', marker='o')
sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^')
sns.lineplot(data=df[df['Type'] == 'Vowel-Consonant'], x='Time Range', y='Shap Res', label='Vowel-Consonant', marker='o')
# sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x')

# Add labels and title
plt.xlabel('Temporal gap')
plt.ylabel('SHAP Value')
plt.title('Shap value for each type')
plt.legend()
plt.xticks(rotation=45)

# Optionally, save the figure
plt.savefig('adjusted_seaborn_plot_order.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()