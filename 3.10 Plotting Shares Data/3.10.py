import yfinance as yf
import matplotlib.pyplot as plt

# Fetching Data for Multiple Tickers
data = yf.download(["ORCL", "AMZN", "INTC"],
                   start="2025-01-01", end="2025-04-07")
print(data.head())


# Plotting the closing prices of both shares
data['Close'].plot(figsize=(10, 6), title="Closing Prices of ORCL and AMZN")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(["Amazon", "Intel", "Oracle"])
plt.grid()
plt.show()
