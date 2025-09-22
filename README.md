# Ames Housing Price Analysis in R
###Project Overview
This project conducts a comprehensive statistical analysis of the Ames Housing dataset using R. The primary goal is to model and understand the relationship between a house's above-ground living area and its sale price. The analysis moves beyond basic regression to employ a suite of advanced statistical computing techniques, demonstrating a deep dive into the data's structure and distribution.

## Key Methods & Skills Showcased
This script demonstrates proficiency in a wide range of statistical methods:

- Least Squares Regression: Calculated "from scratch" using matrix algebra to establish a baseline model.

- Multiple Quantile Regression: A robust method to model different parts of the price distribution (e.g., the 10th, 50th, and 90th percentiles).

- Linear Programming: An alternative optimization approach to solve for the median regression line.

- Maximum Likelihood Estimation (MLE): Fitting and comparing multiple parametric distributions (Weibull, Log-Normal, Gamma) to the sale price data.

- Kernel Density Estimation (KDE): A flexible, non-parametric method to accurately model the true shape of the data's distribution.

- Monte Carlo Simulation: Generating a realistic synthetic dataset based on the superior KDE fit.

## How to Run
1. Prerequisites: You need R and RStudio (recommended).

2. Open the Script: Open the ames_housing_analysis.R script in R or RStudio.

3. Run: Execute the entire script. It will automatically install the required packages (AmesHousing, lpSolve, quantreg, flexsurv).

## Output
Running the script will:

1. Print a detailed analysis and model summaries to the R console.

2. Generate and save four publication-quality plots to your working directory:

  - plot1_ols_fit.png: The baseline OLS regression fit.

  - plot2_quantile_fits_simple.png: Robust quantile regression fits.

  -  plot3_mle_fit_comparison.png: A comparison of parametric and non-parametric distribution fits.

  - plot4_simulation_comparison_kde.png: A comparison of the original data vs. data simulated from the KDE model.
