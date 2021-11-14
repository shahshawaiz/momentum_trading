# %%
# requirments
# !pip install pandas numpy pandas_datareader

# %%
import os
from dateutil import relativedelta, rrule
from datetime import datetime, date, timedelta
import math
import sys
import logging

import pandas as pd
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas.tseries.offsets import MonthEnd, MonthBegin
import pandas_datareader.data as web

# %%
# const vars
INTERVAL_FORMULATION = 12 # formulation period
INTERVAL_HOLDING = 3 # holding period
INTERVAL_INVESTMENT = 1 # investment period
TOP_N_PERCENT = 0.1 # portfolio top % selection range
BOTTOM_N_PERCENT = 0.1 # portfolio bottom % selection range
DEFAULT_WINNER_WEIGHT = 0.5 # default winning weight for classical momentum trading
DEFAULT_LOOSER_WEIGHT = 0.5 # default loosing weight for classical momentum trading
OPTIMIZED_WINNER_WEIGHT = 0.8 # winning weight for optimized momentum trading
OPTIMIZED_LOOSER_WEIGHT = 0.2 # loosing weight for optimized momentum trading
PROCESS_TESTING = True

# dates year-month-date
DATE_SIMULATION_START = datetime(1998, 1, 1)
DATE_SIMULATION_END = datetime(2021, 12, 1)
WEIGHT_OPTIMIZATION_START_DATES = [
    (pd.to_datetime(datetime(2002, 9, 1)), pd.to_datetime(datetime(2004, 2, 1))),
    (pd.to_datetime(datetime(2008, 10, 1)), pd.to_datetime(datetime(2010, 10, 1))),
    (pd.to_datetime(datetime(2011, 9, 1)), pd.to_datetime(datetime(2013, 2, 1))),
    (pd.to_datetime(datetime(2020, 4, 1)), pd.to_datetime(datetime.now()))    
]

# paths
FILENAME_DATA_TESTING = "cleaned_data_export_testing.csv"
FILENAME_DATA_IMPLEMENTATION = "cleaned_data_export_implementation.csv"
PATH_DATA = os.path.join(
    "data"
)
PATH_EXPORTS = os.path.join(
    "exports"
)
PATH_LOG = os.path.join(
    PATH_EXPORTS,
    "log_exec.log"
)
PATH_LOG_HOLDING = os.path.join(
    PATH_EXPORTS,
    "log_holding_generation"
)
PATH_SOURCE_CSV = os.path.join(
    PATH_DATA,
    FILENAME_DATA_TESTING if PROCESS_TESTING else FILENAME_DATA_IMPLEMENTATION
)
PATH_EXPORT_DAILY_RET = os.path.join(
    PATH_EXPORTS,
    "1. daily_returns.csv"
)
PATH_EXPORT_MONTHLY_RET = os.path.join(
    PATH_EXPORTS,
    "2. monthly_returns.csv"
)
PATH_EXPORT_FORMULATION_RET = os.path.join(
    PATH_EXPORTS,
    "3. formulation_returns.csv"
)
PATH_EXPORT_RANKS = os.path.join(
    PATH_EXPORTS,
    "4. ranks.csv"
)
PATH_EXPORT_HOLDING_WINNERS = os.path.join(
    PATH_EXPORTS,
    "4a. holding_winners.csv"
)
PATH_EXPORT_HOLDING_LOOSERS = os.path.join(
    PATH_EXPORTS,
    "4b. holding_loosers.csv"
)
PATH_EXPORT_HOLDING_CLASSIFICAL = os.path.join(
    PATH_EXPORTS,
    "5a. holding_portfolio_classical.csv"
)
PATH_EXPORT_HOLDING_OPTIMIZED = os.path.join(
    PATH_EXPORTS,
    "5b. holding_portfolio_optimized.csv"
)
PATH_EXPORT_HOLDING_YEARLY_CLASSIFICAL = os.path.join(
    PATH_EXPORTS,
    "6a. holding_portfolio_yearly_classical.csv"
)
PATH_EXPORT_HOLDING_YEARLY_OPTIMIZED = os.path.join(
    PATH_EXPORTS,
    "6b.holding_portfolio_yearly_optimized.csv"
)

# logs and exports
TICKER_FILTER_LIST = ["CERG", "JEF", "ALJJ", "APF", "PIY", "KINS", "ROCM", "PETS", "HHY", "FFCO"]
ENABLE_TICKER_FILTER = False
COLUMNS_MONTHLY_HOLDING_RETURNS = ["portfolio", "holding_interval", "year_start", "month_start", "weight_optimized", "winner_returns", "looser_returns", "net_returns", "winner_ticker", "looser_ticker"]
ENABLE_LOGGING = True
LOG_FORMATTER =  '%(asctime)s,%(levelname)s,%(message)s'

# init vars
INIT_STOCK_SUMMARY = {
    "stock_count": 0,
    "current_value": 0
}

def write_to_console(message, log_text=None):
    sys.stdout.write(message + "\n")
    sys.stdout.flush()

    #
    if ENABLE_LOGGING:
        logging.info(message)

# %%
def request_data(ticker, request_start_date, request_end_date, download=False):
    # init df
    res_df = None

    # request datareader
    try:
        res_df = web.DataReader(ticker, 'yahoo', request_start_date, request_end_date)

        #
        if download:
            res_df.to_csv("data/" + ticker + "-" + str(request_start_date) + "-" + request_end_date + ".csv", index=False)
    except:
        print("Error:", ticker, request_start_date, request_end_date)

    return res_df    

def get_market_daily_closing(ticker, daily_date_start):
    #
    daily_returns = 0

    #
    daily_date_end = daily_date_start + timedelta(days=1)

    # request data
    market_daily_stats_df = request_data(ticker, daily_date_start, daily_date_end)

    if market_daily_stats_df is not None:
        # R(i,t)
        daily_returns = 1 + (market_daily_stats_df['Close'] - market_daily_stats_df['Open'])
    
    #
    return daily_returns

def get_market_monthly_closing(ticker, start_date, end_date):
    #
    stock_returns_monthly = 0

    for month_start_date in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):
        #
        month_end_date = month_start_date + relativedelta.relativedelta(months=1, day=1)

        #
        stock_returns_daily = 0
        
        for daily_date in rrule.rrule(rrule.DAILY, dtstart=month_start_date, until=month_end_date):
            #
            stock_returns_daily += get_market_daily_closing(ticker, daily_date)

        #
        stock_returns_monthly += stock_returns_daily

        print("returns monthly:" , stock_returns_monthly, "\n")
            
# 
def get_market_stats_from_internet(ticker_list, start_date, end_date):
    #
    market_stats = dict()

    #
    for ticker in ticker_list:
        #
        get_market_monthly_closing(ticker, start_date, end_date)

    return market_stats

# %%
def read_data(path=PATH_SOURCE_CSV, rows=1000000):
    write_to_console("Reading data from: " + path)

    if ENABLE_TICKER_FILTER:
        return pd.read_csv(path, nrows=rows)    

    return pd.read_csv(path)

def preprocess_data(data_df):
    write_to_console("Preprocessing data")

    # format dt
    data_df['date'] = pd.to_datetime(data_df['datadate'])

    # sort
    data_df.sort_values(by=['tic', 'date'], inplace=True)

    # set index
    # data_df.index = data_df['date']
    
    # extract month year
    data_df['year'] = data_df['date'].dt.year
    data_df['month'] = data_df['date'].dt.month
    data_df['month_startdate'] = data_df['date'].to_numpy().astype('datetime64[M]')

    # use fukter for debugging
    if ENABLE_TICKER_FILTER:
        data_df = data_df.loc[data_df['tic'].isin(TICKER_FILTER_LIST)]

    return data_df   

def init_stock_summary_dict(ticker):
    return dict({ticker: INIT_STOCK_SUMMARY})
# 
def get_portfolio_dict(market_revenue_stats):
    #
    stock_list = market_revenue_stats.tic.unique()

    #
    init_portfolio_dict = list(map(init_stock_summary_dict, stock_list))

    return init_portfolio_dict, stock_list

def get_iteration_intervals(start_date, end_date, interval_months=INTERVAL_INVESTMENT):
    write_to_console("Getting iteration intervals")

    #
    intervals = []

    #
    interval_ticks = pd.date_range(start=start_date - pd.DateOffset(months=INTERVAL_INVESTMENT), end=end_date, freq=pd.offsets.MonthEnd(interval_months))

    #
    for idx, tick in enumerate(interval_ticks):
        if idx > len(interval_ticks) - 2:
            continue

        intervals.append((
            interval_ticks[idx] - MonthBegin(0),
            interval_ticks[idx+1]   
        ))

    return intervals

def get_aggregate_month_plus_one(revenues):    
    revenue_pro = None
    for revenue in revenues:
        if revenue > 0:
            if revenue_pro is None:
                revenue_pro = (1 + revenue) 
            else:
                revenue_pro *= (1 + revenue)

    if revenue_pro is None:
        return 0
        
    return revenue_pro - 1

def get_aggregate_month(revenues):    
    revenue_pro = None
    for revenue in revenues:
        if revenue > 0:
            if revenue_pro is None:
                revenue_pro = revenue
            else:
                revenue_pro *= revenue

    if revenue_pro is None:
        return 0
        
    return revenue_pro - 1

def get_avg_portfolio_return(portfolio_returns):
    return portfolio_returns.mean()

def calculate_monthly_returns(raw_data_df):
    processed_df = raw_data_df.copy()
    processed_df["monthly_return"] = raw_data_df.groupby(["tic", "month_startdate"]).daily_return.transform(get_aggregate_month)
    processed_df = processed_df.drop_duplicates(subset=["tic", 'month_startdate'])
    # processed_df.index = processed_df["month_startdate"]
    
    return processed_df

def calculate_formulation_returns(raw_data_df):
    write_to_console("Calculating formulation returns")

    processed_df = raw_data_df.copy()
    formulation_returns = calculate_returns(processed_df, INTERVAL_FORMULATION,  "tic", "monthly_return", "formulation_returns")

    return formulation_returns
    
def calculate_holding_returns(raw_data_df):
    processed_df = raw_data_df.copy()
    holding_returns = calculate_returns(processed_df, INTERVAL_HOLDING,  "tic", "monthly_return", "holding_returns")

    return holding_returns

def calculate_returns(raw_data_df, period, group_columns, apply_column, new_column_name):
    monthly_grouped_df = raw_data_df.copy()
    monthly_grouped_df[new_column_name] = monthly_grouped_df.groupby(group_columns)[apply_column].transform(lambda s: s.rolling(period).apply(get_aggregate_month))

    return monthly_grouped_df

def calculate_avg_holding_returns(raw_data_df, period, group_columns, apply_column, new_column_name):
    avg_holding_df = raw_data_df.copy()

    avg_holding_df = calculate_returns(avg_holding_df, period, group_columns, apply_column, "holding_returns")
    avg_holding_df[new_column_name] = avg_holding_df.groupby(group_columns)["holding_returns"].transform(get_avg_portfolio_return)

    return avg_holding_df

def get_holding_avg_returns(raw_data_df, tic_list, interval_start, interval_end, return_weight=1):
    processed_df = raw_data_df.copy()
    filtered_df = processed_df[processed_df['month_startdate'].between(interval_start, interval_end, inclusive='both')]
    filtered_df = filtered_df[filtered_df['tic'].isin(tic_list)] 

    #
    holding_df = calculate_avg_holding_returns(filtered_df, INTERVAL_HOLDING,  "tic", "monthly_return", "avg_holding_returns")

    return holding_df.avg_holding_returns.mean() * return_weight

def get_winner_looser_list(raw_data_df, interval_start, interval_end, winner_looser_type):
    write_to_console("Getting portfolio for {} to {}".format(interval_start, interval_end))

    processed_df = raw_data_df.copy()
    filtered_df = processed_df[processed_df['month_startdate'].between(interval_start, interval_end, inclusive='both')]    

    #
    tic_filetered = filtered_df[winner_looser_type]

    if not tic_filetered.empty:
        tic_list = []

        #
        for interval_tics_list in tic_filetered:
            interval_tics_list = interval_tics_list.split(",")
            for tic in interval_tics_list:
                if tic not in tic_list:
                    tic_list.append(tic)

        return tic_list

    return []

def get_ranking_by_interval(raw_data_df):
    write_to_console("Getting formulation ranking")
    
    processed_df = raw_data_df.copy()

    # get ranks
    processed_df["compiled_rank"] = processed_df.groupby(["month_startdate"]).formulation_returns.transform(pd.DataFrame.rank, ascending=False)
    processed_df = processed_df.sort_values(["year", "month", "compiled_rank"], ascending=True)

    # drop nans
    processed_df = processed_df[processed_df['compiled_rank'].notna()]

    # get top and bottom n rank ranges
    ranks_range = processed_df["compiled_rank"].unique() 
    ranks_count = len(ranks_range)
    top_n_ranks = ranks_range[:math.ceil(ranks_count * TOP_N_PERCENT)]
    bottom_n_ranks = ranks_range[-math.ceil(ranks_count * BOTTOM_N_PERCENT):]
    
    # assign winner/looser label
    processed_df.loc[processed_df.compiled_rank.isin(top_n_ranks), 'status'] = 'w'
    processed_df.loc[processed_df.compiled_rank.isin(bottom_n_ranks), 'status'] = 'l'    

    #
    winner_df = processed_df.loc[processed_df['status'] == 'w']
    winner_df["winners"] = winner_df.groupby(["month_startdate"]).tic.transform(lambda x: ",".join(x.astype(str)))
    # winner_df = winner_df.drop_duplicates(subset=['month_startdate'])

    looser_df = processed_df.loc[processed_df['status'] == 'l']
    looser_df["loosers"] = looser_df.groupby(["month_startdate"]).tic.transform(lambda x: ",".join(x.astype(str)))
    # looser_df = looser_df.drop_duplicates(subset=['month_startdate'])

    return processed_df, winner_df, looser_df

def check_crisis_period_enabled(interval_start, interval_end):
    for optimized_start, optimized_end in WEIGHT_OPTIMIZATION_START_DATES: 
        if optimized_start <= interval_start and optimized_end >= interval_end:
            return True
    return False

def get_yearly_holding_returns(raw_data_df):
    processed_df = raw_data_df.copy()
    yearly_holding_df = processed_df.groupby(["year_start"]).agg({
        "winner_returns": "mean", 
        "looser_returns": "mean",
        "net_returns": "mean",	
    })
    
    return yearly_holding_df

def get_holding_interval(investment_interval_start, investment_interval_end):
    holding_interval_start = investment_interval_start - pd.DateOffset(months=INTERVAL_HOLDING) + MonthBegin(0)  
    holding_interval_end = investment_interval_start - pd.DateOffset(months=1) + MonthEnd(0)

    #
    if INTERVAL_INVESTMENT > INTERVAL_HOLDING:
        holding_interval_end = investment_interval_start - pd.DateOffset(months=INTERVAL_INVESTMENT) + MonthBegin(0)
    
    return holding_interval_start, holding_interval_end

def export_portfolio_holdings(start_date, end_date, monthly_df, winner_df, looser_df, export_filenames, is_optimized=False):

    # init portfolio weights
    WRITE_SKIP_COUNT = 15
    weight_winning, weight_loosing = DEFAULT_WINNER_WEIGHT, DEFAULT_LOOSER_WEIGHT
    crisis_period_enabled = False

    # for_formulation_ranking
    iteration_intervals = get_iteration_intervals(start_date, end_date, INTERVAL_INVESTMENT)

    # init dfs
    monthly_holding_df = pd.DataFrame(columns=COLUMNS_MONTHLY_HOLDING_RETURNS)
    yearly_holding_df = pd.DataFrame(columns=COLUMNS_MONTHLY_HOLDING_RETURNS)

    #
    portfolio_count = 1

    # iterate over 3 months
    for investment_interval_start, investment_interval_end in iteration_intervals:
        #
        holding_interval_start, holding_interval_end = get_holding_interval(investment_interval_start, investment_interval_end)

        # init vars
        year, month, month_end = investment_interval_start.year, investment_interval_start.month, (investment_interval_start + pd.Timedelta(INTERVAL_INVESTMENT, unit='m')).month
        investment_interval = "{}-{}".format(str(investment_interval_start), str(investment_interval_end))
        holding_interval = "{}-{}".format(str(holding_interval_start), str(holding_interval_end))        
        winner_returns = 0
        looser_returns = 0
        portfolio_id = "Portfolio {}".format(portfolio_count - WRITE_SKIP_COUNT)


        # check if optimized
        if is_optimized:
            # check if within optimized weight period
            crisis_period_enabled = check_crisis_period_enabled(investment_interval_start, investment_interval_end)

            #
            if crisis_period_enabled:
                weight_winning, weight_loosing = OPTIMIZED_WINNER_WEIGHT, OPTIMIZED_LOOSER_WEIGHT
            else:
                weight_winning, weight_loosing = DEFAULT_WINNER_WEIGHT, DEFAULT_LOOSER_WEIGHT
        
        #
        winners_list = get_winner_looser_list(winner_df, holding_interval_start, holding_interval_end, "winners")

        if len(winners_list) != 0:
            winner_returns = get_holding_avg_returns(monthly_df, winners_list, holding_interval_start, holding_interval_end, weight_winning)
            log_str = '{},{},{},{}'.format(year, month, ";".join(winners_list), winner_returns)

            write_to_console("Winner found :{}".format(log_str), log_str)
            write_to_console("Winner returns: {}".format(winner_returns))

        #
        loosers_list = get_winner_looser_list(looser_df, holding_interval_start, holding_interval_end, "loosers")

        if len(loosers_list) != 0:
            looser_returns = get_holding_avg_returns(monthly_df, loosers_list, holding_interval_start, holding_interval_end, weight_loosing)
            log_str = '{},{},{},{}'.format(year, month, ";".join(loosers_list), looser_returns)

            write_to_console("Looser found :{}".format(log_str), log_str)
            write_to_console("Looser returns: {}".format(looser_returns))

        net_returns = winner_returns - looser_returns
        
        row_df = pd.DataFrame([[ portfolio_id, holding_interval, year, month, crisis_period_enabled, winner_returns, looser_returns, net_returns, winners_list, loosers_list]], columns=COLUMNS_MONTHLY_HOLDING_RETURNS)

        #
        monthly_holding_df = pd.concat(
            [monthly_holding_df, row_df],
            ignore_index = True
        )

        # increment portfolio count
        portfolio_count += 1

    monthly_holding_df = monthly_holding_df[WRITE_SKIP_COUNT:] 

    #
    monthly_holding_df.to_csv(export_filenames[0])

    # get yearly holding returns
    yearly_holding_df = get_yearly_holding_returns(monthly_holding_df)
    yearly_holding_df.to_csv(export_filenames[1])

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(LOG_FORMATTER)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def init_logging():
    logging.basicConfig(filename=PATH_LOG, format=LOG_FORMATTER, level=logging.INFO)

    #
    write_to_console("Init Logs")

# %%
#
def run_simulation(start_date, end_date):
    # 
    init_logging()

    # read
    data_df = read_data()

    # preprocess data
    data_df = preprocess_data(data_df)
    data_df.to_csv(PATH_EXPORT_DAILY_RET)

    # calculate monthly returns
    monthly_df = calculate_monthly_returns(data_df)
    monthly_df.to_csv(PATH_EXPORT_MONTHLY_RET)

    # calculate formulation returns
    formulation_df = calculate_formulation_returns(monthly_df)
    formulation_df.to_csv(PATH_EXPORT_FORMULATION_RET)

    # get ranking
    ranked_df, winner_df, looser_df = get_ranking_by_interval(formulation_df)        
    ranked_df.to_csv(PATH_EXPORT_RANKS)

    # # calculate holding winning
    # holding_df_winner = calculate_holding_returns(winner_df)
    # holding_df_winner.rename(columns={"avg_holding_returns": "winning_returns"}, inplace=True)
    # holding_df_winner.to_csv(PATH_EXPORT_HOLDING_WINNERS)

    # # calculate holding returns
    # holding_df_looser = calculate_holding_returns(looser_df)
    # holding_df_looser.rename(columns={"avg_holding_returns": "winning_returns"}, inplace=True)    
    # holding_df_looser.to_csv(PATH_EXPORT_HOLDING_LOOSERS)

    # # #
    # # combined_df = holding_df_winner.merge(holding_df_looser, on="month_startdate", suffixes=("_winner", "_looser"))
    # # combined_df.to_csv(PATH_EXPORT_HOLDING_RET)

    # export portfolio holdings
    export_portfolio_holdings(start_date, end_date, monthly_df, winner_df, looser_df, [
        PATH_EXPORT_HOLDING_CLASSIFICAL,
        PATH_EXPORT_HOLDING_YEARLY_CLASSIFICAL
    ])
    export_portfolio_holdings(start_date, end_date, monthly_df, winner_df, looser_df, [
        PATH_EXPORT_HOLDING_OPTIMIZED,
        PATH_EXPORT_HOLDING_YEARLY_OPTIMIZED
    ], True)

    # break
    # update portfolio: stock-wise capital allocation
    # print(rankings.head())

# %%
# run monthly simulation using market return stats
run_simulation(DATE_SIMULATION_START, DATE_SIMULATION_END)


