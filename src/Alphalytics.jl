module Alphalytics

using TimeSeries
using DataFrames
using StatsBase
using Plots
using GR


export rows_spearmanr, rows_pearsonr, spearman_factor_decay, mean_autocor, rolling_mean_autocor,
    plot_factor_distribution, plot_performance_table,

    quantile_return, compute_quantiles_returns, 
    quantile_turnover, quantiles_turnover, total_quantiles_turnover, quantile_performance_table, quantile_chg,

    row_mean, column_mean, rowwise_ordinalrank, rowwise_competerank, rowwise_tiedrank, rowwise_denserank,
    rowwise_ordinal_pctrank, rowwise_tied_pctrank, 
    rowwise_quantiles, rowwise_tiedquantiles, rowwise_count, rowwise_countall,

    cumulative_growth, pct_change, drawdowns, annual_return, annual_stdev, 
    annual_sharpe_ratio, downside_deviation, sortino_ratio, max_drawdown, 
    down_capture, up_capture, overall_capture, performance_table

# Write your package code here.

    include("ts_utils.jl")
    include("performance_analytics.jl")

    include("ic_analytics.jl")
    include("quantile_analytics.jl")
    
    include("plotting.jl")
    

end
