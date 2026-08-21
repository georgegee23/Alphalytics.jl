########################################## Alphalytics Quantile Analysis ###########################################################


####### QUANTILE ANALYSIS ##########################################################

"""
    quantile_return(quantiles_ts::TimeArray, returns_ts::TimeArray, quantile::Any) -> TimeArray

Calculate the equally weighted mean return of the securities sitting in one quantile.

For each timestamp the securities assigned to `quantile` are selected and their returns
averaged, giving the return series of a portfolio that holds that quantile.

# Arguments
- `quantiles_ts::TimeArray`: Quantile assignments, one column per security. These must
    already be lagged so that a period's assignment lines up with the return it earns;
    pairing an assignment with the return of the same period looks ahead.
- `returns_ts::TimeArray`: Realised returns, same shape and timestamps as `quantiles_ts`.
- `quantile::Any`: The quantile to select, matched against the values of `quantiles_ts`.

# Returns
- `TimeArray`: A single column named `Q<quantile>` holding the mean return per timestamp.
    A period with no securities in `quantile` is NaN.

# Throws
- `AssertionError`: If the two TimeArrays do not have matching timestamps.
"""
function quantile_return(quantiles_ts::TimeArray, returns_ts::TimeArray, quantile::Any)

    @assert timestamp(quantiles_ts) == timestamp(returns_ts) "TimeArrays must have matching timestamps"
    
    quantiles_mtx = values(quantiles_ts)
    returns_mtx = values(returns_ts)
    
    function quantile_rowmean(q_row, r_row, qtile) # row mean of non-nan values
        relevant_returns = r_row[q_row .== qtile]
        if isempty(relevant_returns)
            return NaN  # or another appropriate value to indicate no data
        else
            return mean(filter(!isnan, relevant_returns))
        end
    end
    
    quantile_rets_vec = [quantile_rowmean(q_row, r_row, quantile) for (q_row, r_row) in zip(eachrow(quantiles_mtx), eachrow(returns_mtx))]
    
    return TimeArray(timestamp(quantiles_ts), quantile_rets_vec, ["Q$quantile"])
end

"""
    compute_quantiles_returns(quantiles_ta::TimeArray, returns_ta::TimeArray, n_quantiles::Int) -> TimeArray

Build the return series of every quantile portfolio at once.

This applies [`quantile_return`](@ref) to each quantile from 1 to `n_quantiles` and merges
the results, giving the return panel that the rest of the quantile analysis consumes.

# Arguments
- `quantiles_ta::TimeArray`: Quantile assignments, one column per security, already lagged
    to line up with the returns they earn.
- `returns_ta::TimeArray`: Realised returns, same shape and timestamps as `quantiles_ta`.
- `n_quantiles::Int`: Number of quantiles to compute.

# Returns
- `TimeArray`: One column per quantile, named `Q1` through `Q<n_quantiles>`.

# Throws
- `AssertionError`: If the two TimeArrays do not have matching timestamps.

# Example
```julia
quantiles = rowwise_tied_quantiles(factor_scores, 5)
qrets = compute_quantiles_returns(lag(quantiles, 1), forward_returns, 5)
```
"""
function compute_quantiles_returns(quantiles_ta::TimeArray, returns_ta::TimeArray, n_quantiles::Int)

    @assert timestamp(quantiles_ta) == timestamp(returns_ta) "TimeArrays must have matching timestamps"

    # Pre-allocate array to store results
    results = Vector{TimeArray}(undef, n_quantiles)

    # Compute returns for each quantile
    for q in 1:n_quantiles
        results[q] = quantile_return(quantiles_ta, returns_ta, q)
    end

    # Merge all results into a single TimeArray
    quan_rets_ts = merge(results...)

    return quan_rets_ts
end


"""
    quantile_turnover(quantile_ta::TimeArray, target_quantile::Number) -> TimeArray

Calculate how much of a single quantile's membership is replaced each period.

A security counts as unchanged when it sat in `target_quantile` last period and still does
this period. Turnover is the fraction of the current members that are new arrivals, so 0.0
means the bucket is unchanged and 1.0 means it turned over completely.

# Arguments
- `quantile_ta::TimeArray`: Quantile assignments, one column per security, ordered by time.
- `target_quantile::Number`: The quantile whose turnover is measured.

# Returns
- `TimeArray` with three columns, starting one period after the input because the first
    period has nothing to compare against:
    - `Turnover`: Fraction of the quantile's members that changed from the previous period.
    - `NoChange`: Count of members carried over from the previous period.
    - `Total`: Count of securities in the quantile this period.

# Throws
- `ArgumentError`: If `quantile_ta` has fewer than 2 time points.

# Notes
- A period in which no security sits in `target_quantile` divides by zero and yields NaN
    for that period's turnover.
"""
function quantile_turnover(quantile_ta::TimeArray, target_quantile::Number)

    if size(quantile_ta, 1) < 2
        throw(ArgumentError("TimeArray must have at least 2 time points"))
    end

    # Filtering for the target quantile, excluding first row to align with lag
    q_target_ta = (quantile_ta .== target_quantile)[2:end]

    # Lag the quantile TimeArray by one period
    quantile_ta_lag = lag(quantile_ta, 1)

    # Remove the first row to match lengths with the lagged TimeArray
    quantile_ta = quantile_ta[2:end]

    # Identify where there's no change for specified quantile target
    no_change = (quantile_ta .== quantile_ta_lag) .& q_target_ta

    # Sum up no changes for each time period
    no_change_count = sum(no_change, dims=2)

    # Count total valid entries per time period at the target quantile
    total_securities = sum(q_target_ta, dims=2)

    # Calculate turnover
    turnover_ta = 1 .- (no_change_count ./ total_securities)
    turnover_ta = hcat(turnover_ta, no_change_count, total_securities)
    turnover_ta = TimeSeries.rename(turnover_ta, ["Turnover", "NoChange", "Total"])

    return turnover_ta
end

"""
    quantiles_turnover(quantile_ta::TimeArray) -> TimeArray

Calculate the turnover of every quantile, side by side.

This runs [`quantile_turnover`](@ref) for each quantile from 1 up to the largest one present
in the data and keeps only the `Turnover` column of each, so the quantiles can be compared
directly. Extreme quantiles usually turn over faster than middle ones.

# Arguments
- `quantile_ta::TimeArray`: Quantile assignments, one column per security, ordered by time.

# Returns
- `TimeArray`: One column per quantile, named `Q1` through `Q<max>`, holding that quantile's
    turnover fraction. Starts one period after the input.

# Throws
- `ArgumentError`: If `quantile_ta` has fewer than 2 time points.
"""
function quantiles_turnover(quantile_ta::TimeArray)

    if size(quantile_ta, 1) < 2
        throw(ArgumentError("TimeArray must have at least 2 time points"))
    end
    
    max_quantile = values(quantile_ta) |> x -> filter(!isnan, x) |> maximum
    turnover_ta = reduce(merge, [quantile_turnover(quantile_ta, q)["Turnover"] for q in 1:max_quantile])

    quantile_names = ["Q"*string(Int(i)) for i in 1:max_quantile]
    turnover_ta = TimeSeries.rename(turnover_ta, quantile_names)
    return turnover_ta
end

"""
    total_quantiles_turnover(quantile_ta::TimeArray) -> TimeArray

Calculate turnover across the whole universe rather than within one quantile.

A security counts as unchanged when it stays in the same quantile from one period to the
next, whichever quantile that is. This is the single headline number for how much the factor
reshuffles its rankings, and so for the trading it implies.

# Arguments
- `quantile_ta::TimeArray`: Quantile assignments, one column per security, ordered by time.

# Returns
- `TimeArray` with three columns, starting one period after the input:
    - `Turnover`: Fraction of securities that changed quantile since the previous period.
    - `NoChange`: Count of securities that stayed in the same quantile.
    - `Total`: Count of securities with a valid (non-NaN) assignment this period.

# Throws
- `ArgumentError`: If `quantile_ta` has fewer than 2 time points.

# Notes
- Turnover is 1 minus the ratio of unchanged securities to the total with valid entries,
    so a period in which every security is NaN yields NaN.
"""
function total_quantiles_turnover(quantile_ta::TimeArray)

    if size(quantile_ta, 1) < 2
        throw(ArgumentError("TimeArray must have at least 2 time points"))
    end

    # Lag the quantile TimeArray by one period
    quantile_ta_lag = lag(quantile_ta, 1)

    # Remove the first row to match lengths with the lagged TimeArray
    quantile_ta = quantile_ta[2:end]

    # Identify where there's no change
    no_change = quantile_ta .== quantile_ta_lag

    # Sum up changes for each time period
    no_change_count = sum(no_change, dims=2)

    # Count total valid entries per time period
    total_securities = sum(.!isnan.(quantile_ta), dims=2)

    # Calculate turnover
    turnover_ta = 1 .- no_change_count ./ total_securities
    turnover_ta = hcat(turnover_ta, no_change_count, total_securities)
    turnover_ta = TimeSeries.rename(turnover_ta, ["Turnover", "NoChange", "Total"])

    return turnover_ta
end


"""
    quantile_performance_table(quantile_returns::TimeArray, benchmark_returns::TimeArray;
                               thresh_value::Number = 0, periods_per_year::Int) -> DataFrame

Compute table with summary performance statistics for factor quantiles.

# Arguments
- `quantile_returns::TimeArray`: Returns for each quantile, one column per quantile.
- `benchmark_returns::TimeArray`: Benchmark returns, aligned to `quantile_returns`.
- `thresh_value::Number`: Threshold separating up from down markets, also used as the
    minimum acceptable return of the Sortino ratio (default: 0).
- `periods_per_year::Int`: Number of return periods in a year (e.g. 52 for weekly data).

# Returns
- `DataFrame` with one row per statistic and one column per quantile, plus a leading
    `Stat` column naming each row.

# Details
- The return, standard deviation and drawdown rows are reported in percent. The Sharpe,
    Sortino and capture rows are ratios and are left on their natural scale.
"""
function quantile_performance_table(quantile_returns::TimeArray, benchmark_returns::TimeArray; thresh_value::Number = 0, periods_per_year::Int)

    @assert size(quantile_returns, 1) == size(benchmark_returns,1) "Quantile returns and benchmark row counts do not match."

    quantile_names = colnames(quantile_returns)

    # Percentage-valued statistics, reported in percent.
    annual_returns = annualized_return(quantile_returns, periods_per_year) .* 100
    annual_std = annualized_stdev(quantile_returns, periods_per_year) .* 100
    max_dds = max_drawdown(quantile_returns) .* 100

    # Ratios, left on their natural scale.
    sharpe_ratios = annualized_sharpe_ratio(quantile_returns, periods_per_year)
    sortino_ratios = sortino_ratio(quantile_returns, mar = thresh_value)
    dc_ratio = down_capture(quantile_returns, benchmark_returns, thresh_value)
    uc_ratio = up_capture(quantile_returns, benchmark_returns, thresh_value)
    oc_ratio = overall_capture(quantile_returns, benchmark_returns, thresh_value)

    stats_names = ["Annual Return", "Annual StDev", "Sharpe Ratio", "Sortino Ratio", "Max Drawdowns",
    "Down Capture", "Up Capture", "Overall Capture"]

    summary_stats_table = DataFrame([annual_returns, annual_std, sharpe_ratios, sortino_ratios, max_dds,
    dc_ratio, uc_ratio, oc_ratio], :auto)
    summary_stats_table = permutedims(summary_stats_table)
    summary_stats_table = DataFrames.rename(summary_stats_table, quantile_names)


    summary_stats_table[!, "Stat"] = stats_names
    summary_stats_table = select(summary_stats_table, :Stat, quantile_names...)

    return summary_stats_table

end






"""
    quantile_chg(perf_table::DataFrame) -> DataFrame

Measure how steadily each statistic changes as you step from one quantile to the next.

A factor that works should improve monotonically across quantiles, so the step-to-step change
in each statistic is taken, then summarised by its mean, its standard deviation, and their
ratio. A large positive ratio means the statistic climbs consistently rather than erratically.

# Arguments
- `perf_table::DataFrame`: A table as returned by [`quantile_performance_table`](@ref):
    a leading `Stat` column naming each row, then one column per quantile in order.

# Returns
- `DataFrame` with one row per statistic and the columns `Stat`, `MeanChg`, `StdChg`
    and `Ratio` (`MeanChg / StdChg`).

!!! warning "Not currently runnable"
    This function calls `shift_dataframe`, which is not defined in Alphalytics or in any of
    its dependencies, and refers to an undefined `q_perf_table` on its last few lines. Both
    need resolving before it can be called.
"""
function quantile_chg(perf_table::DataFrame)

    tb = permutedims(perf_table)

    tb = tb[2:end,:] 
    tb = DataFrames.rename(tb, perf_table.Stat)

    vecs_chg = tb .- shift_dataframe(tb, shift = 1) |> eachcol .|> col -> filter(!ismissing, col)

    mu_chg =  vecs_chg .|> mean
    std_chg = vecs_chg .|> std
    ratio_chg = mu_chg ./ std_chg

    results = DataFrame(:MeanChg => mu_chg, :StdChg => std_chg, :Ratio => ratio_chg, :Stat => q_perf_table.Stat)
    results = results[:, [:Stat, :MeanChg, :StdChg, :Ratio]]

    return results

end