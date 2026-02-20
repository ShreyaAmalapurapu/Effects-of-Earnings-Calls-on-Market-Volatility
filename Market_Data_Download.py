import databento as db
import pandas as pd

company = pd.read_csv('top20_sp500_tech_companies.csv')
company['识别号'] = company['交易所'].apply(lambda x: 'XNAS.ITCH' if x.upper() == 'NASDAQ' else 'XNYS.PILLAR') # XNAS.ITCH = Nasdaq; XNYS.PILLAR = NYSE

header_mapping = {
    '排名': 'Rank',
    '公司名称': 'Company',
    '股票代码': 'Ticker',
    '交易所': 'Exchange',
    '识别号': 'Exchange_2'
}
company = company.rename(columns=header_mapping)

exchange = "ENTER HERE" # XNAS.ITCH = Nasdaq; XNYS.PILLAR = NYSE
ticker = "ENTER HERE" # ticker name
callname = "ENTER HERE"
utc_time_str = "ENTER HERE" # earning call start time in UTC

utc_time = pd.Timestamp(utc_time_str) # change string to timedelta
start_utc = utc_time - pd.Timedelta(hours=1)
end_utc = utc_time + pd.Timedelta(hours=2)

API_KEY = "ENTER HERE" # enter your API key
client = db.Historical(API_KEY)

cost = client.metadata.get_cost( # check the cost (we only have $125 in total)
    dataset=exchange,
    schema="ohlcv-1s",
    symbols=[ticker], # Company name
    start= start_utc,  # UTC Time
    end= end_utc
)
print("The cost will be",cost) # after we find it's coverable, continue the next step

data = client.timeseries.get_range(
    dataset=exchange,
    schema="ohlcv-1s",
    symbols=[ticker], # Company name
    start=start_utc,  # UTC Time
    end=end_utc
)
df = data.to_df()
print(df.head())

# filename = f"{ticker}_{callname}.parquet"
# df.to_parquet(filename)
filename2 = f"{ticker}_{callname}.csv" # save the file
df.to_csv(filename2)
