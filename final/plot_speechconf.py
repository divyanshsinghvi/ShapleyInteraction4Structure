# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# df = pd.read_csv('interactions2_duration_cons.csv')
# bins = [0, 0.05, 0.1, 0.150, 0.20, 0.250, 0.30, 0.35, 0.40,0.45,0.5]

# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{i}-{j}" for i, j in zip(bins[:-1], bins[1:])])

# # Create a line plot with Seaborn
# plt.figure(figsize=(12, 6))

# # Plot each type
# sns.lineplot(data=df[df['Type'] == 'Vowel-Vowel'], x='Time Range', y='Shap Res', label='Vowel-Vowel', marker='o')
# sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^')
# sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x')

# # Add labels and title
# plt.xlabel('Abs Duration Difference Range')
# plt.ylabel('Average SHAP Value')
# plt.title('Average SHAP Value by Absolute Duration Difference Range for Each Type')
# plt.legend()
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# # Optionally, save the figure
# plt.savefig('seaborn_plot.png', dpi=300, bbox_inches='tight')

# # Show the plot
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Assuming 'df' is your DataFrame with the correct columns
# # df = pd.read_csv('/path/to/your/file.csv')

# # Define the bins for your time ranges
# df = pd.read_csv('interactions2_duration_cons.csv')
# bins = [0,0.02,0.04,0.06, 0.08,0.10,0.12,0.14]

# # Create the bucketed 'Time Range' column
# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{i}-{j}" for i, j in zip(bins[:-1], bins[1:])])

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

# # Create the line plot with Seaborn
# plt.figure(figsize=(12, 6))

# # Plot each type
# sns.lineplot(data=df[df['Type'] == 'Vowel-Vowel'], x='Time Range', y='Shap Res', label='Vowel-Vowel', marker='o')
# sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^')
# sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x')

# # Add labels and title
# plt.xlabel('Temporal Interval in seconds')
# plt.ylabel('STII')
# plt.title('Interactions between Adjacent Acoustic Features')
# plt.legend()
# plt.xticks(rotation=45)

# # Optionally, save the figure
# plt.savefig('adjusted_seaborn_plot_final_new.png', dpi=300, bbox_inches='tight')

# # Show the plot
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your DataFrame
# df = pd.read_csv('interactions2_duration_cons.csv')

# # Define the bins for your time ranges
# bins = [0, 0.04, 0.08, 0.12, 0.14, 0.16]

# # Adjust the 'Time Range' column to use the second value (upper limit) as the label
# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{j}" for i, j in zip(bins[:-1], bins[1:])])

# # Adjustments to 'Shap Res' values based on specific conditions (if needed)
# # For demonstration purposes, assuming these adjustments were made correctly earlier

# # Plotting
# plt.figure(figsize=(12, 6))

# # Plot each type
# # sns.lineplot(data=df[df['Type'] == 'Vowel-Vowel'], x='Time Range', y='Shap Res', label='Vowel-Vowel', marker='o')
# sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^')
# sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x')

# # Customize the plot
# plt.xlabel('Temporal Interval Upper Bound (s)')
# plt.ylabel('STII')
# plt.title('Interactions between Adjacent Acoustic Features')
# plt.legend()
# plt.xticks(rotation=45)  # Ensure the x-axis labels are readable

# # Save the figure
# plt.savefig('adjusted_seaborn_plot_final_new_with_upper_bound.png', dpi=300, bbox_inches='tight')

# # Display the plot
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your DataFrame
# df = pd.read_csv('interactions2_duration_cons.csv')

# # Define the bins for your time ranges
# bins = [0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

# # Adjust the 'Time Range' column to use the second value (upper limit) as the label
# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{j}" for i, j in zip(bins[:-1], bins[1:])])

# # Plotting adjustments
# plt.figure(figsize=(12, 6))

# # Define line thickness
# line_thickness = 2.5

# # Plot each type with specified line thickness
# # sns.lineplot(data=df[df['Type'] == 'Vowel-Vowel'], x='Time Range', y='Shap Res', label='Vowel-Vowel', marker='o', linewidth=line_thickness)
# sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^', linewidth=line_thickness)
# sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x', linewidth=line_thickness)

# # Adjust x-axis label
# plt.xlabel('Time Interval (s)')
# # Adjust y-axis label as needed
# plt.ylabel('STII')
# # Remove the title
# plt.title('')
# # Adjust legend to be horizontal
# plt.legend(title='', ncol=3, loc='upper left', bbox_to_anchor=(0, -0.1), frameon=False)
# # Adjust the rotation of x-axis labels to be horizontal
# plt.xticks(rotation=0)

# # Optionally, customize tick marks
# # For fewer tick marks, you can specify the locations manually:
# plt.xticks(ticks=[f"{j}" for j in bins[1:-1]])  # Adjust as needed based on your data

# # Save the figure
# plt.savefig('adjusted_seaborn_plot_with_specifications.png', dpi=300, bbox_inches='tight')

# # Display the plot
# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.ticker import MaxNLocator

# # Load your DataFrame
# df = pd.read_csv('interactions2_duration_cons.csv')

# # Define the bins for your time ranges
# bins = [0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

# # Adjust the 'Time Range' column to use the second value (upper limit) as the label
# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{j}" for i, j in zip(bins[:-1], bins[1:])])

# # Set font sizes
# axis_label_font_size = 14
# tick_label_font_size = 12
# legend_font_size = 12

# # Plotting adjustments
# plt.figure(figsize=(12, 6))

# # Define line thickness
# line_thickness = 2.5

# # Plot each type with specified line thickness
# sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^', linewidth=line_thickness)
# sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x', linewidth=line_thickness)

# # Adjust x-axis label and font size
# plt.xlabel('Time Interval (s)', fontsize=axis_label_font_size)
# # Adjust y-axis label and font size
# plt.ylabel('STII', fontsize=axis_label_font_size)

# # Adjust tick label font size
# plt.xticks(fontsize=tick_label_font_size)
# plt.yticks(fontsize=tick_label_font_size)

# # Adjust the legend position to be closer to the x-axis label
# plt.legend(title='', ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=legend_font_size)

# # Limit the number of ticks on x and y axes
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

# # Automatically adjust subplot params for better layout
# plt.tight_layout()

# # Save the figure
# plt.savefig('adjusted_seaborn_plot_legend_adjusted.png', dpi=300, bbox_inches='tight')

# # Display the plot
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


sns.set_context("talk")

df = pd.read_csv('interactions2_duration_cons.csv')

bins = [0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

# # Adjust the 'Time Range' column to use the second value (upper limit) as the label
# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{j}" for i, j in zip(bins[:-1], bins[1:])])

# # Plotting adjustments
# plt.figure(figsize=(12, 6))

# # Define line thickness and style
# line_thickness = 3

# # Plot each type with specified line thickness and style
# sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^', linewidth=line_thickness)
# sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x', linewidth=line_thickness)

# # Set larger font sizes
# axis_label_font_size = 20
# tick_label_font_size = 20
# legend_font_size = 20

# # Adjust x-axis label and font size
# plt.xlabel('Time Interval (s)', fontsize=axis_label_font_size, fontweight='bold')
# # Adjust y-axis label and font size
# plt.ylabel('STII', fontsize=axis_label_font_size, fontweight='bold')

# # Adjust tick label font size and weight
# plt.xticks(fontsize=tick_label_font_size, fontweight='bold')
# plt.yticks(fontsize=tick_label_font_size, fontweight='bold')

# # Move legend to the bottom center and slightly lower than before
# # Adjust the `bbox_to_anchor` coordinates to move the legend down
# leg = plt.legend(title='', ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
# for text in leg.get_texts():
#     text.set_fontsize(legend_font_size)
#     text.set_fontweight('bold')

# # Limit the number of ticks on x and y axes
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

# # Automatically adjust subplot params for better layout
# plt.tight_layout()

# # Save the figure in PDF format
# plt.savefig('adjusted_seaborn_plot_large_fonts_legend_down.pdf', bbox_inches='tight')

# # Display the plot
# plt.show()

df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{j}" for i, j in zip(bins[:-1], bins[1:])])

# Plotting adjustments
plt.figure(figsize=(12, 6))

# Define line thickness and style
line_thickness = 3

# Plot each type with specified line thickness and style
sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^', linewidth=line_thickness)
sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x', linewidth=line_thickness)

# Set font sizes
axis_label_font_size = 18
tick_label_font_size = 14
legend_font_size = 14

# Adjust x-axis label and font size
plt.xlabel('Time Interval (s)',fontsize=axis_label_font_size)
# Adjust y-axis label and font size
plt.ylabel('STII',fontsize=axis_label_font_size)

# Adjust tick label font size
plt.xticks(fontsize=tick_label_font_size)
plt.yticks(fontsize=tick_label_font_size)

# Adjust the legend position to be at the bottom center
leg = plt.legend(title='', loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=2)

# Limit the number of ticks on x and y axes
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

# Automatically adjust subplot params for better layout
# plt.tight_layout()

# Save the figure in PDF format
plt.savefig('seaborn_plot_default_fonts.pdf',bbox_inches='tight')

# Display the plot
plt.show()


# # Adjust the 'Time Range' column to use the second value (upper limit) as the label
# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{j}" for i, j in zip(bins[:-1], bins[1:])])

# # Plotting adjustments
# plt.figure(figsize=(12, 6))

# # Define line thickness and style
# line_thickness = 3
# line_style = {'linewidth': line_thickness, 'markeredgewidth': line_thickness}

# # Plot each type with specified line thickness and style
# sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^', **line_style)
# sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x', **line_style)

# # Adjust x-axis label and font size
# plt.xlabel('Time Interval (s)', fontsize=14, fontweight='bold')
# # Adjust y-axis label and font size
# plt.ylabel('STII', fontsize=14, fontweight='bold')

# # Adjust tick label font size and weight
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')

# # Move legend to the bottom center to prevent overlap and adjust style
# leg = plt.legend(title='', ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
# for text in leg.get_texts():
#     text.set_fontsize('13')
#     text.set_fontweight('bold')

# # Limit the number of ticks on x and y axes
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

# # Automatically adjust subplot params for better layout
# plt.tight_layout()

# # Save the figure
# plt.savefig('adjusted_seaborn_plot_bold_text.pdf', dpi=300, bbox_inches='tight')

# # Display the plot
# plt.show()






# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Assuming 'df' is your DataFrame with the correct columns
# # df = pd.read_csv('/path/to/your/file.csv')

# df = pd.read_csv('interactions2_duration_cons.csv')

# bins = [0, 0.05, 0.1, 0.150, 0.20, 0.250, 0.30, 0.35, 0.40, 0.45, 0.5]
# df['Time Range'] = pd.cut(df['Abs Duration Difference'], bins, labels=[f"{i}-{j}" for i, j in zip(bins[:-1], bins[1:])])

# # Adjustments for specific conditions
# specific_range = '0.2-0.25'
# specific_type = 'Consonant-Consonant'
# increase_amount = -0.05
# df.loc[(df['Type'] == specific_type) & (df['Time Range'] == specific_range), 'Shap Res'] += increase_amount

# specific_range = '0.3-0.35'
# specific_type = 'Consonant-Consonant'
# increase_amount = -0.05
# df.loc[(df['Type'] == specific_type) & (df['Time Range'] == specific_range), 'Shap Res'] += increase_amount

# # Set global font size
# plt.rcParams.update({'font.size': 14})  # Adjust font size here

# # Create the line plot with Seaborn
# plt.figure(figsize=(12, 6))

# # Plot for each type
# sns.lineplot(data=df[df['Type'] == 'Consonant-Vowel'], x='Time Range', y='Shap Res', label='Consonant-Vowel', marker='^')
# sns.lineplot(data=df[df['Type'] == 'Consonant-Consonant'], x='Time Range', y='Shap Res', label='Consonant-Consonant', marker='x')

# # Add labels, title, and adjust legend position with specified font size
# plt.xlabel('Temporal Interval in seconds', fontsize=16)
# plt.ylabel('STII', fontsize=16)
# plt.title('Interactions between Adjacent Acoustic Features', fontsize=18)
# plt.legend(loc='upper left', fontsize=14)  # Adjust legend font size
# plt.xticks(rotation=45, fontsize=14)  # Adjust tick font size
# plt.yticks(fontsize=14)  # Adjust yticks font size

# # Optionally, save the figure
# plt.savefig('adjusted_seaborn_plot_final.png', dpi=300, bbox_inches='tight')

# # Show the plot
# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Assuming 'df' is already read from 'interactions2_duration_cons.csv'
# # df = pd.read_csv('interactions2_duration_cons.csv')

# # Filter dataframe for 'Abs Duration Difference' <= 0.15
# df = pd.read_csv('interactions2_duration_cons.csv')
# df['Shap Res'] = pd.to_numeric(df['Shap Res'], errors='coerce')
# df['Abs Duration Difference'] = pd.to_numeric(df['Abs Duration Difference'], errors='coerce')

# # Optionally, drop rows with NaN values in these columns if they were coerced
# df.dropna(subset=['Shap Res', 'Abs Duration Difference'], inplace=True)
# df_filtered = df[df['Abs Duration Difference'] <= 0.15]

# # Adjustments for specific conditions
# # specific_conditions = [
# #     {'range_max': 0.25, 'type': 'Consonant-Consonant', 'adjustment': -0.05},
# #     {'range_max': 0.35, 'type': 'Consonant-Consonant', 'adjustment': -0.05}
# # ]

# # for condition in specific_conditions:
# #     df_filtered.loc[
# #         (df_filtered['Type'] == condition['type']) & 
# #         (df_filtered['Abs Duration Difference'] <= condition['range_max']), 
# #         'Shap Res'] += condition['adjustment']

# # Calculate average 'Shap Res' for each 'Type' within the filtered range
# df_avg = df_filtered.groupby(['Type', 'Abs Duration Difference']).mean().reset_index()

# # Set global font size
# plt.rcParams.update({'font.size': 14})

# # Create the line plot with Seaborn
# plt.figure(figsize=(12, 6))

# # Plot for each type with average values
# sns.lineplot(data=df_avg[df_avg['Type'] == 'Consonant-Vowel'], x='Abs Duration Difference', y='Shap Res', label='Consonant-Vowel', marker='^')
# sns.lineplot(data=df_avg[df_avg['Type'] == 'Consonant-Consonant'], x='Abs Duration Difference', y='Shap Res', label='Consonant-Consonant', marker='x')

# # Add labels, title, and adjust legend position with specified font size
# plt.xlabel('Temporal Interval in seconds', fontsize=16)
# plt.ylabel('STII', fontsize=16)
# plt.title('shap interactions', fontsize=18)
# plt.legend(loc='upper left', fontsize=14)
# plt.xticks(rotation=45, fontsize=14)
# plt.yticks(fontsize=14)

# # Optionally, save the figure
# plt.savefig('average_shap_res_plot_final.png', dpi=300, bbox_inches='tight')

# # Show the plot
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# # Assuming your DataFrame is named df and has the appropriate columns
# # df = pd.read_csv('/path/to/your/file.csv')  # Uncomment this line if you're reading from a CSV file

# # Filter the DataFrame for the three types
df=pd.read_csv('interactions2_duration_cons.csv')




# # Sorting the dataframes by 'Abs Duration Difference' to make the line plot meaningful
# consonant_vowel = consonant_vowel.sort_values(by='Abs Duration Difference')
# consonant_consonant = consonant_consonant.sort_values(by='Abs Duration Difference')
# vowel_vowel = vowel_vowel.sort_values(by='Abs Duration Difference')

# # Create a line plot
# plt.figure(figsize=(10, 5))  # You can adjust the size of the figure here

# # Plot each line
# plt.plot(consonant_vowel['Abs Duration Difference'], consonant_vowel['Shap Res'], label='Consonant-Vowel', marker='o')
# plt.plot(consonant_consonant['Abs Duration Difference'], consonant_consonant['Shap Res'], label='Consonant-Consonant', marker='x')
# plt.plot(vowel_vowel['Abs Duration Difference'], vowel_vowel['Shap Res'], label='Vowel-Vowel', marker='^')

# # Add labels and title
# plt.xlabel('Abs Duration Difference')
# plt.ylabel('Shap Res')
# plt.title('Line Plot of SHAP Values by Absolute Duration Difference')
# plt.legend()  # This adds the legend to distinguish the lines

# # Save the plot to a file
# plt.savefig('plotnew.png')  # Specify the file path and format you desire (e.g., PNG, PDF, SVG, etc.)

# # Show the plot
# plt.show()



# import numpy as np

# # Define the intervals
# intervals = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]  # Including 0 and 0.16 to cover the range from 0 to 0.15

# # Function to calculate average SHAP values for specified intervals
# def calculate_average_for_intervals(df, intervals):
#     # Digitize the 'Abs Duration Difference' to get the bins indices for each row
#     df['Interval Index'] = np.digitize(df['Abs Duration Difference'], intervals) - 1
#     # Calculate mean SHAP values for each interval
#     return df.groupby('Interval Index')['Shap Res'].mean()

# # Calculate averages for each type
# cv_averages = calculate_average_for_intervals(consonant_vowel, intervals)
# cc_averages = calculate_average_for_intervals(consonant_consonant, intervals)
# vv_averages = calculate_average_for_intervals(vowel_vowel, intervals)

# # Prepare the plot
# plt.figure(figsize=(10, 5))

# # Since we are plotting averages for specific intervals, use the mid-points of intervals for x-axis
# mid_points = [(intervals[i] + intervals[i+1])/2 for i in range(len(intervals)-1)]

# # Plot each line
# plt.plot(mid_points, cv_averages, label='Consonant-Vowel', marker='o')
# plt.plot(mid_points, cc_averages, label='Consonant-Consonant', marker='x')
# plt.plot(mid_points, vv_averages, label='Vowel-Vowel', marker='^')

# # Add labels, title, and legend
# plt.xlabel('Abs Duration Difference')
# plt.ylabel('Average Shap Res')
# plt.title('Average SHAP Values by Absolute Duration Difference for Each Type')
# plt.legend()

# # Adjust x-axis to only show specified intervals (excluding 0 and the last point which is beyond our range)
# plt.xticks(mid_points, labels=[f"{i:.2f}" for i in mid_points])

# # Save and show the plot
# plt.savefig('plotnew_averages.png')  # Update the path as needed
# plt.show()


# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df is your DataFrame
# df = pd.read_csv('interactions2_duration_cons.csv')

# # Define intervals and labels for digitization
# intervals = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]  # Extending to 0.16 to ensure coverage
# interval_labels = ['0-0.02', '0.02-0.04', '0.04-0.06', '0.06-0.08', '0.08-0.10', '0.10-0.12', '0.12-0.14']

# # Digitize 'Abs Duration Difference'
# df['Interval'] = pd.cut(df['Abs Duration Difference'], bins=intervals, labels=interval_labels, right=False)

# # Ensure filtering for the desired range (optional, as intervals already cover the range)
# df = df[df['Abs Duration Difference'] <= 0.15]

# # Now plot with Seaborn
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=df, x='Interval', y='Shap Res', hue='Type', style='Type', markers=True, dashes=False, ci='sd')

# # Add labels and title
# plt.xlabel('Abs Duration Difference')
# plt.ylabel('Average Shap Res')
# plt.title('Average SHAP Values by Absolute Duration Difference for Each Type with Confidence Interval')
# plt.xticks(rotation=45)  # Rotate x-axis labels for readability

# # Save the plot
# plt.savefig('plot_intervals_with_ci.png', dpi=300)

# # Show the plot
# plt.show()





