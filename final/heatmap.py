import pandas as pd


consonants = ["P", "T", "K", "CH", "F", "TH", "S", "SH", "HH", "B", "D", "G", "JH", "V", "DH", "Z", "ZH", "M", "N", "NG", "L", "W", "R", "Y", "VLS", "VOI", "SON", "GLI", "LIQ"]


df = pd.read_csv('interactions2_duration_cons_order.csv')


print(df.head())


import matplotlib.pyplot as plt

import pandas as pd

# Assuming df is your DataFrame
df = pd.read_csv('interactions2_duration_cons_order.csv')

# Inspect the first few rows of the DataFrame
print(df.head())

# Check unique values in 'Type' column
print("Unique Types:", df['Type'].unique())

# Filter and calculate the average interaction
filtered_df = df[(df['Type'] == 'Vowel-Consonant') & (df['Abs Duration Difference'] >= 0) & (df['Abs Duration Difference'] <= 0.1)]
average_interaction = filtered_df.groupby('Phoneme To')['Shap Res'].mean()

# Check if the filtered DataFrame is empty
print(average_interaction)

sorted_average_interaction = average_interaction.sort_values()

# Create a bar plot with sorted values
sorted_average_interaction.plot(kind='bar', figsize=(10, 6))

# Add labels and title
plt.xlabel('Phoneme To')
plt.ylabel('Average SHAP Value')
plt.title('Average SHAP Value for Vowel-Consonant in 0 to 0.1 Time Range (Sorted)')

# Optionally, rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Save the plot to a file
plt.savefig('sorted_average_interaction_bar_chart.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


