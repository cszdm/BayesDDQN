import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt  
import scipy.stats as stats  

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams['axes.labelsize'] = 16  
plt.rcParams['xtick.labelsize'] = 14  
plt.rcParams['ytick.labelsize'] = 14 

# Sample data  
sample_data = [0.025, 0.031, 0.027, 0.031, 0.027, 0.030, 0.028, 0.03, 0.026, 0.028]  
# sample_data = [0.027, 0.031, 0.027, 0.031, 0.027, 0.030, 0.028, 0.03, 0.026, 0.028]


# Calculate sample mean  
mean_savings = np.mean(sample_data)  
print(f"Sample Mean Energy Savings: {mean_savings:.2%}")  
  
# Perform t-test (single-sample t-test) but essentially calculate the confidence interval  
n = len(sample_data)  # Sample size  
se = stats.sem(sample_data)  # Standard error  
confidence = 0.95  # 95% confidence level  
h = se * stats.t.ppf((1 + confidence) / 2., n - 1)  # Calculate half-width of the confidence interval  
  
# Calculate confidence interval  
ci_lower = mean_savings - h  
ci_upper = mean_savings + h  
print(f"95% Confidence Interval for Energy Savings: {ci_lower:.2%} to {ci_upper:.2%}")  
  
# Plot KDE with confidence interval and mean  
kde_data = stats.gaussian_kde(sample_data)  
x = np.linspace(min(sample_data), max(sample_data), 1000)  
plt.plot(x, kde_data(x), color='blue', label='KDE of Energy Savings')  
plt.axvline(mean_savings, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_savings:.2%}')  
plt.axvline(ci_lower, color='g', linestyle='--', linewidth=2, label=f'95% CI Lower: {ci_lower:.2%}')  
plt.axvline(ci_upper, color='g', linestyle='--', linewidth=2, label=f'95% CI Upper: {ci_upper:.2%}')  
# plt.title('KDE of Energy Savings with Mean and 95% Confidence Interval')  
plt.xlabel('Energy Savings (%)')  
plt.ylabel('Kernel Density Estimation')  
plt.legend()  
plt.grid(True)  
plt.show()
