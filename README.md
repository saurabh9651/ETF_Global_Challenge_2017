# ETF Global Challenge 2017 Portfolio Optimization
This repository contains the code for the ETF Global Challenge competition of 2017. The goal of this project is to create a portfolio optimization strategy that will maximize the Sharpe Ratio and minimize the portfolio variance. The code utilizes three different optimization methods: Minimum Variance Optimization, Maximum Sharpe Ratio Optimization, and Equal Weighted Optimization. The performance of each method is then compared based on their out-of-sample Sharpe Ratios.

## Features
- Portfolio optimization using three different methods
- Out-of-sample Sharpe Ratio comparison
- Visualization of portfolio performance
- Calculation of the last day weights and units to be purchased

## Dependencies
```
numpy
pandas
datetime
matplotlib
pandas-datareader
quandl
statsmodels
scipy
```

## Installation
1. Clone the repository or download the code as a zip file.
```
git clone https://github.com/username/etf-global-challenge-2017.git
```
2. Install the required dependencies using pip.
```
pip install -r requirements.txt
```
3. Run the Python script in the terminal or any Python environment.
```
python ETF_SPRING_2017.py
```

## Usage
The script fetches historical stock data for selected tickers and performs portfolio optimization using three different methods:
1. Minimum Variance Optimization (MINV)
2. Maximum Sharpe Ratio Optimization (MSR)
3. Equal Weighted Optimization (EQW)

The script then calculates the out-of-sample Sharpe Ratio for each method and visualizes the portfolio performance. The last day weights and units to be purchased are also calculated and displayed for each optimization method.

## Results
The script outputs the portfolio performance visualization, out-of-sample Sharpe Ratios for each optimization method, and the last day weights and units to be purchased.

## Contributing
Feel free to submit pull requests, report issues, or provide suggestions to improve the code. Your contributions are welcome!

## License
This project is licensed under the MIT License. See LICENSE for more information.
