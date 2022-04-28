# bi-level-framework
Traffic Incident Duration prediction using bi-level framework

The code explores scenarios of traffic incident duration prediction task using data available for San-Francisco area from 2016 to 2021 (as an excerpt from Countrywise Traffic accident data set by Sobhan Moosavi, https://smoosavi.org/datasets/us_accidents). Victoria roads, Sydney and M7 Motorway, Sydney are not placed here since data sets are private.
There are two version of SF data set: a) SF.csv - excerpt from original data set with zero-variance fields removed (e.g. "State" variable is always equal to one value, therefore ommited), b) SF2.csv - excerpt from original data with digitized fields.

1. Code to reproduce ECDF and histograms (figure 1 in original paper) in the form of jupyter notebooks can be found in the 0_PROFILING folder.
2. Distribution of incident durations according to MUTCD classifcation (figure 3): 7_EXTENSION/mutcd.ipynb
3. Incident duration classification using varying thresholds (figure 7) and Low-duration outliers (LDO) (figure 8) scenarios are in 1_CLASSIFICATION folder.
4. Regression extrapolation (tables 3-5) and I/E joint optimization (tables 6-8) scenarios: 2_REGRESSION folder.
5. Multi-class classification matrix (figure 9): multi.ipynb in 7_EXTENSION folder
6. Regression using Quantiled Time Folding (figures 10-11): qunatiled time folding.ipynb in 7_EXTENSION folder
7. Code to obtain SHAP values for regression experiments:
	- All-to-All SHAP value estimation is in folder 2_REGRESSION (figure 14) and SHAP analysis specific to short-term/long-term incidents (figures 15-17) is in folder 7_EXTENSION
8. Comparison of a fusion and single model performance: 5_FULLMODEL folder.



