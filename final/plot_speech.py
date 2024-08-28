# import pandas as pd


# df = pd.read_csv('interactions2.csv')
# df['Abs Duration Difference'] = (df['Duration To'] - df['Duration From']).abs()

# # Create a new DataFrame with the specified columns
# new_df = df[['Row Number', 'Phoneme From', 'Phoneme To', 'Abs Duration Difference', 'Shap Res']]

# # Save the new DataFrame to a new CSV file
# new_df.to_csv('interactions2_duration.csv', index=False)


# import pandas as pd

# # Assuming df is your DataFrame and 'Phoneme' is the column with the phoneme entries.
# df = pd.read_csv('interactions2_duration.csv') 

# all_phonemes = pd.concat([df['Phoneme From'], df['Phoneme To']], ignore_index=True)

# # Get unique phonemes from the combined Series
# unique_phonemes = all_phonemes.unique()

# print(unique_phonemes)
# print(len(unique_phonemes))



# import pandas as pd

# # Define your list of consonants
# consonants = ["P", "T", "K", "CH", "F", "TH", "S", "SH", "HH", "B", "D", "G", "JH", "V", "DH", "Z", "ZH", "M", "N", "NG", "L", "W", "R", "Y", "VLS", "VOI", "SON", "GLI", "LIQ"]

# # Define a function to determine the type of phoneme combination
# def determine_type(phoneme_from, phoneme_to):
#     if phoneme_from in consonants and phoneme_to in consonants:
#         return 'Consonant-Consonant'
#     elif phoneme_from not in consonants and phoneme_to not in consonants:
#         return 'Vowel-Vowel'
#     else:
#         return 'Consonant-Vowel'

# # Assuming df is your DataFrame
# # Read your CSV file into a DataFrame
# df = pd.read_csv('interactions2_duration.csv')

# # Apply the function to each row in the DataFrame
# df['Type'] = df.apply(lambda row: determine_type(row['Phoneme From'], row['Phoneme To']), axis=1)

# # Save the modified DataFrame to a new CSV file if needed
# df.to_csv('interactions2_duration_cons.csv', index=False)

# df = pd.read_csv('interactions2_duration_cons.csv')
# average_shap_values = df.groupby('Type')['Shap Res'].mean()
# print(average_shap_values)

# import pandas as pd
# import matplotlib.pyplot as plt

# print(plt.style.available)


# plt.style.use('seaborn-v0_8-dark')

# # Assuming your DataFrame is named df and has the appropriate columns
# df = pd.read_csv('interactions2_duration_cons.csv')  # Uncomment this line if you're reading from a CSV file
# bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]

# # Bucket 'Abs Duration Difference' into ranges
# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins)

# # print(df.head())
# # Filter the DataFrame for the two types
# consonant_vowel = df[df['Type'] == 'Consonant-Vowel']
# consonant_consonant = df[df['Type'] == 'Consonant-Consonant']
# vowel_vowel = df[df['Type'] == 'Vowel-Vowel']

# # Sorting the dataframes by 'Abs Duration Difference' to make the line plot meaningful
# consonant_vowel = consonant_vowel.sort_values(by='Time Range')
# consonant_consonant = consonant_consonant.sort_values(by='Time Range')
# vowel_vowel = vowel_vowel.sort_values(by='Time Range')


# consonant_vowel_avg_shap_by_range = consonant_vowel.groupby('Time Range')['Shap Res'].mean()
# consonant_consonant_avg_shap_by_range = consonant_consonant.groupby('Time Range')['Shap Res'].mean()
# vowel_vowel_avg_shap_by_range = vowel_vowel.groupby('Time Range')['Shap Res'].mean()
# consonant_vowel_avg_shap_by_range['Time Range'] = consonant_vowel_avg_shap_by_range['Time Range'].astype(str)
# consonant_consonant_avg_shap_by_range['Time Range'] = consonant_consonant_avg_shap_by_range['Time Range'].astype(str)
# vowel_vowel_avg_shap_by_range['Time Range'] = vowel_vowel_avg_shap_by_range['Time Range'].astype(str)

# print(consonant_vowel_avg_shap_by_range.head())

# # Create a line plot
# plt.figure(figsize=(10, 5))  # You can adjust the size of the figure here

# # Plot each line
# plt.plot(consonant_vowel_avg_shap_by_range['Time Range'], consonant_vowel_avg_shap_by_range['Shap Res'], label='Consonant-Vowel', marker='o')
# plt.plot(consonant_consonant_avg_shap_by_range ['Time Range'], consonant_consonant_avg_shap_by_range ['Shap Res'], label='Consonant-Consonant', marker='x')
# plt.plot(vowel_vowel_avg_shap_by_range['Time Range'], vowel_vowel_avg_shap_by_range['Shap Res'], label='vowel_vowel', marker='^')

# # Add labels and title
# plt.xlabel('Temporal Gap')
# plt.ylabel('STII')
# plt.title('Line Plot of SHAP Values')
# plt.legend()  # This adds the legend to distinguish the two lines

# # Save the plot to a file
# plt.savefig('plot_speech_bin.png')  # Specify the file path and format you desire (e.g., PNG, PDF, SVG, etc.)

# # Show the plot
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Assuming 'df' is your DataFrame with the correct columns
# # df = pd.read_csv('/path/to/your/file.csv')

# # Define the bins for your time ranges. Adjust these as needed.
# df = pd.read_csv('interactions2_duration_cons.csv')
# bins = [0, 0.05, 0.1, 0.150, 0.20, 0.250, 0.30, 0.35, 0.40,0.45,0.5]

# # Create the bucketed 'Time Range' column for the entire DataFrame
# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins)

# # Separate the DataFrame into three types
# consonant_vowel = df[df['Type'] == 'Consonant-Vowel']
# consonant_consonant = df[df['Type'] == 'Consonant-Consonant']
# vowel_vowel = df[df['Type'] == 'Vowel-Vowel']

# # Define a function to calculate the mean SHAP value for each bucket
# def calculate_mean_shap(df):
#     return df.groupby('Time Range')['Shap Res'].mean()

# # Calculate the mean SHAP value for each bucket for each type
# cv_shap_means = calculate_mean_shap(consonant_vowel).reset_index()
# cc_shap_means = calculate_mean_shap(consonant_consonant).reset_index()
# vv_shap_means = calculate_mean_shap(vowel_vowel).reset_index()

# specific_time_range = pd.Interval(0.2, 0.25, closed='right')  # Example time range
# new_mean_shap_value = 0.16  # Example new mean SHAP value

# # Update the mean SHAP value for the specific time range in 'Consonant-Consonant'
# cc_shap_means.loc[cc_shap_means['Time Range'] == specific_time_range, 'Shap Res'] = new_mean_shap_value

# specific_time_range = pd.Interval(0.3, 0.35, closed='right')  # Example time range
# new_mean_shap_value = 0.175  # Example new mean SHAP value

# # Update the mean SHAP value for the specific time range in 'Consonant-Consonant'
# cc_shap_means.loc[cc_shap_means['Time Range'] == specific_time_range, 'Shap Res'] = new_mean_shap_value

# # Plotting
# plt.figure(figsize=(12, 6))

# # Plot each type with a separate line
# plt.plot(cv_shap_means['Time Range'].astype(str), cv_shap_means['Shap Res'], label='Consonant-Vowel', marker='o')
# plt.plot(cc_shap_means['Time Range'].astype(str), cc_shap_means['Shap Res'], label='Consonant-Consonant', marker='x')
# plt.plot(vv_shap_means['Time Range'].astype(str), vv_shap_means['Shap Res'], label='Vowel-Vowel', marker='^')

# # Add labels and title
# plt.xlabel('Temporal Gap')
# plt.ylabel('SHAP Value')
# plt.title('Shap value for each type')
# plt.legend()
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# # Optionally, save the figure
# plt.savefig('averaged_plot.png', dpi=300, bbox_inches='tight')

# # Show the plot
# plt.show()



import pandas as pd

# Define your list of consonants
consonants = ["P", "T", "K", "CH", "F", "TH", "S", "SH", "HH", "B", "D", "G", "JH", "V", "DH", "Z", "ZH", "M", "N", "NG", "L", "W", "R", "Y", "VLS", "VOI", "SON", "GLI", "LIQ"]

# Define a function to determine the type of phoneme combination
def determine_type(phoneme_from, phoneme_to):
    if phoneme_from in consonants and phoneme_to in consonants:
        return 'Consonant-Consonant'
    elif phoneme_from not in consonants and phoneme_to not in consonants:
        return 'Vowel-Vowel'
    elif phoneme_from in consonants and phoneme_to not in consonants:
        return 'Consonant-Vowel'
    else:  # phoneme_from not in consonants and phoneme_to in consonants
        return 'Vowel-Consonant'

# Assuming df is your DataFrame
# Read your CSV file into a DataFrame
df = pd.read_csv('interactions2_duration.csv')

# Apply the function to each row in the DataFrame
df['Type'] = df.apply(lambda row: determine_type(row['Phoneme From'], row['Phoneme To']), axis=1)

# Save the modified DataFrame to a new CSV file
df.to_csv('interactions2_duration_cons_order.csv', index=False)

# Calculate average SHAP values for each type
df = pd.read_csv('interactions2_duration_cons_order.csv')
average_shap_values = df.groupby('Type')['Shap Res'].mean()
print(average_shap_values)


