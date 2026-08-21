########################################## Performance Analytics ###########################################################





########################################### RETURN STATS ##########################################

"""
    returns_to_prices(returns::TimeArray, init_value::Int = 1) -> TimeArray

Compute the cumulative product of returns, preserving NaN locations.

This function treats the input as returns (as decimals or percentages), so it adds 1 to each value before multiplying.  
NaN values are preserved in the output, and the cumulative product continues from the last non-NaN value after a NaN.

# Arguments
- `returns::TimeArray`: TimeArray containing return data with possible NaN values.
- `init_value::Int`: Initial value for the cumulative product (default: 1).

# Returns
- `TimeArray`: TimeArray with cumulative products, preserving NaN locations.

# Example
```julia
ta = TimeArray([0.1, NaN, 0.05, -0.02], timestamp=1:4, colnames=[:R])
returns_to_prices(ta)
```
"""
function returns_to_prices(returns::TimeArray, init_value::Int = 1)
    values_matrix = values(returns)
    result = similar(values_matrix)
    for col in 1:size(values_matrix, 2)
        cumulative = init_value
        for row in 1:size(values_matrix, 1)
            if isnan(values_matrix[row, col])
                result[row, col] = NaN
            else
                cumulative *= (1 + values_matrix[row, col])
                result[row, col] = cumulative
            end
        end
    end
    return TimeArray(timestamp(returns), result, colnames(returns))
end


"""
    to_pctchange(prices::TimeArray, window::Int64 = 1) -> TimeArray

Calculate the percentage change of prices over a specified window.

# Arguments
- `prices::TimeArray`: A TimeArray containing price data. Can be single or multi-column.
- `window::Int64`: The size of the window over which to calculate the percentage change.
    Must be at least 1.

# Returns
- `TimeArray`: A new TimeArray with the same timestamps and column names as the input,
    but with values representing the percentage changes.

# Details
- Missing values in the input are replaced with NaN.
- Percentage change is calculated as (current_price / lagged_price) - 1.
- The first `window` rows of the result are NaN, since there is not enough history to
    compute a change for them.
"""
function to_pctchange(prices::TimeArray, window::Int64=1)

    if window < 1
        throw(ArgumentError("Window size must be at least 1"))
    end

    prices = coalesce.(prices, NaN)
    w_prices_lagged = TimeSeries.lag(prices, window, padding = true)
    ts_matrix = (values(prices) ./ values(w_prices_lagged)) .- 1
    col_names = colnames(prices)
    timestps = timestamp(prices)
    pct_change_ta = TimeArray(timestps, ts_matrix, col_names)

    return pct_change_ta
end

#---------------------------------------------------------------------------------

"""
    to_cumulative_growth(returns::TimeArray, init_value::Number = 1) -> TimeArray

Compound a series of returns into a growth curve, preserving NaN locations.

Each value is treated as a return, so 1 is added before multiplying. The result is the value
an initial investment of `init_value` would have grown to by each timestamp.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset, possibly with NaN.
- `init_value::Number`: Starting value of the growth curve (default: 1).

# Returns
- `TimeArray`: Same timestamps and column names, holding the compounded value at each point.

# Details
- NaN inputs stay NaN in the output, and compounding resumes from the last non-NaN value
    rather than restarting, so a gap in the data does not reset the curve.

# Example
```julia
ta = TimeArray([0.1, NaN, 0.05], timestamp = 1:3, colnames = [:R])
to_cumulative_growth(ta)  # 1.1, NaN, 1.155
```
"""
function to_cumulative_growth(returns::TimeArray, init_value::Number = 1)

    values_matrix = values(returns)
    result = similar(values_matrix)

    for col_index in axes(values_matrix, 2)
        cumulative = init_value
        for row_index in axes(values_matrix, 1)
            if isnan(values_matrix[row_index, col_index])
                result[row_index, col_index] = NaN
            else
                cumulative *= (1 + values_matrix[row_index, col_index])
                result[row_index, col_index] = cumulative
            end
        end
    end
  
    return TimeArray(timestamp(returns), result, colnames(returns))
end

"""
    to_cummax(returns::TimeArray) -> TimeArray

Compute the running peak of the growth curve implied by a series of returns.

Note that the input is *returns*, not levels: the returns are compounded with
[`to_cumulative_growth`](@ref) first, then the running maximum is taken. This high-water
mark is what drawdowns are measured against.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset, possibly with NaN.

# Returns
- `TimeArray`: Same timestamps and column names, holding the highest growth value seen up to
    and including each timestamp.

# Details
- NaN inputs stay NaN in the output, and each column starts its running maximum at its own
    first non-NaN entry, so columns with different start dates are handled independently.
"""
function to_cummax(returns::TimeArray)

    ta = to_cumulative_growth(returns)

    vals = values(ta)
    cummax_vals = similar(vals)
    if ndims(vals) == 1
        runningmax = NaN
        for i in 1:length(vals)
            if isnan(vals[i])
                cummax_vals[i] = NaN
            else
                runningmax = isnan(runningmax) ? vals[i] : max(runningmax, vals[i])
                cummax_vals[i] = runningmax
            end
        end
    else
        rows, columns = size(vals)
        for col in 1:columns
            runningmax = NaN
            for row in 1:rows
                if isnan(vals[row, col])
                    cummax_vals[row, col] = NaN
                else
                    runningmax = isnan(runningmax) ? vals[row, col] : max(runningmax, vals[row, col])
                    cummax_vals[row, col] = runningmax
                end
            end
        end
    end
    return TimeArray(timestamp(ta), cummax_vals, colnames(ta))  # Corrected
end


"""
    arithmetic_return(returns::TimeArray) -> Vector{Float64}

Compute the simple average periodic return of each asset.

This is the plain mean of the return series. It overstates realised growth whenever returns
vary, because it ignores compounding; [`geom_return`](@ref) is the compounded counterpart.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.

# Returns
- `Vector{Float64}`: One mean return per column, in column order. NaN returns are ignored.
"""
function arithmetic_return(returns::TimeArray)

    return column_mean(returns)
end


"""
    geom_return(returns::TimeArray) -> Vector{Float64}

Compute the compounded average periodic return of each asset.

This is the constant per-period return that would have produced the same total growth as the
actual series. Unlike [`arithmetic_return`](@ref) it accounts for compounding, so it is the
honest figure to quote for realised performance.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.

# Returns
- `Vector{Float64}`: One geometric mean return per column, in column order. NaN returns are
    ignored, and a column with fewer than 2 valid observations yields NaN.
"""
function geom_return(returns::TimeArray)

    vals = values(returns)
    # Preallocate result vector with length equal to number of columns
    n_cols = size(vals, 2)
    result = Vector{Float64}(undef, n_cols)

    # Iterate over each column
    for col in 1:n_cols
        # Filter non-NaN values from the column
        valid_vals = filter(!isnan, vals[:, col])
        
        # Check if there are at least 2 non-NaN values
        if length(valid_vals) >= 2
            # Calculate geometric mean return
            compounded_growth = prod(1 .+ valid_vals)
            n = length(valid_vals)
            geo_mean_return = compounded_growth^(1/n) - 1
            result[col] = geo_mean_return
        else
            # Set to NaN if fewer than 2 non-NaN values
            result[col] = NaN
        end
    end

    return result
end


"""
    annualized_return(returns::TimeArray, periods_per_year::Int) -> Vector{Float64}

Compute the compound annual growth rate of each asset.

The returns are compounded into a total growth figure, then rescaled to a one-year horizon,
giving the annual rate that would have produced the same result over the sample.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `periods_per_year::Int`: Number of periods in a year (252 daily, 52 weekly, 12 monthly).

# Returns
- `Vector{Float64}`: One annualized return per column, in column order.

# Throws
- `AssertionError`: If `returns` contains fewer than two data points.

# Notes
- The whole sample length is used for every column, even if some columns start later. Drop
    leading NaNs first if you need columns to be compared over identical windows.
"""
function annualized_return(returns::TimeArray, periods_per_year::Int)

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"

    compounded_growth = to_cumulative_growth(returns)
    n_periods = size(compounded_growth, 1)
    ann_rets = (last.(eachcol(values(compounded_growth))) .^ (periods_per_year / n_periods)) .- 1

    return ann_rets

end


"""
    drawdowns(returns::TimeArray) -> TimeArray

Compute the full drawdown series of each asset: how far below its running peak it sits at
every point in time.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.

# Returns
- `TimeArray`: Same timestamps and column names, holding the drawdown at each point as a
    negative decimal, where -0.2 means 20% below the peak. A value of 0.0 means the asset is
    at a new high.

# Example
```julia
dd = drawdowns(returns)
minimum(values(dd))  # the worst point of the deepest drawdown
```
"""
function drawdowns(returns::TimeArray)

    cumgrowth = to_cumulative_growth(returns)
    # Calculate the cumulative maximum of the returns
    dd = cumgrowth ./ to_cummax(returns) .-1
    dd = TimeSeries.rename!(dd, colnames(returns))

    return dd
end


"""
    max_drawdown(returns::TimeArray) -> Vector{Float64}

Compute the deepest peak-to-trough loss each asset suffered over the sample.

This is the single worst point of [`drawdowns`](@ref), and the usual measure of how much pain
holding the asset would have involved.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.

# Returns
- `Vector{Float64}`: One maximum drawdown per column, in column order, as a *positive*
    decimal: 0.35 means the asset fell 35% below its peak at worst. NaN drawdowns are ignored.
"""
function max_drawdown(returns::TimeArray)

    dd = drawdowns(returns)
    abs_dd = abs.(dd)
    vals = values(abs_dd)
    max_vals = [maximum(filter(!isnan, vals[:, col])) for col in 1:size(vals, 2)]
    return max_vals
end



#################################### VOLATILITY STATS ########################################

"""
    stdev(returns::TimeArray) -> Vector{Float64}

Compute the periodic volatility of each asset.

This is the sample standard deviation of the return series, in the same units as the returns
themselves. Use [`annualized_stdev`](@ref) to put it on an annual footing.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.

# Returns
- `Vector{Float64}`: One standard deviation per column, in column order. NaN returns are
    ignored, and a column with fewer than 2 valid observations yields NaN.

# Throws
- `AssertionError`: If `returns` has fewer than two data points, or holds non-numeric data.
"""
function stdev(returns::TimeArray)

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert all(eltype(values(returns)) <: Real) "All columns must contain numeric data"
    
    vals = values(returns)
    std_dev = [length(filter(!isnan, vals[:, col])) >= 2 ? std(filter(!isnan, vals[:, col])) : NaN for col in 1:size(vals, 2)]
    return std_dev
end


"""
    annualized_stdev(returns::TimeArray, periods_per_year::Int) -> Vector{Float64}

Compute the annualized volatility of each asset.

The periodic standard deviation from [`stdev`](@ref) is scaled by the square root of
`periods_per_year`, the standard convention for putting volatility on an annual footing.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `periods_per_year::Int`: Number of periods in a year (252 daily, 52 weekly, 12 monthly).

# Returns
- `Vector{Float64}`: One annualized standard deviation per column, in column order.

# Throws
- `AssertionError`: If `returns` contains fewer than two data points.

# Notes
- The square-root-of-time rule assumes returns are serially uncorrelated. Trending or
    mean-reverting series will have their true annual volatility under- or overstated.
"""
function annualized_stdev(returns::TimeArray, periods_per_year::Int)

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    
    std_dev = stdev(returns)
    # Convert to a vector for easier manipulation
    annual_std_dev = std_dev .* sqrt(periods_per_year)
    return annual_std_dev
end


"""
    downside_deviation(returns::TimeArray, mar::Number = 0; corrected::Bool = true) -> Vector{Float64}

Compute the volatility of the losses only, ignoring upside moves.

Plain standard deviation punishes large gains as much as large losses. Downside deviation
looks solely at returns falling below the minimum acceptable return, which matches how
investors actually experience risk. It is the denominator of the Sortino ratio.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `mar::Number`: Minimum acceptable return; only returns below this count (default: 0).
- `corrected::Bool`: Apply Bessel's correction, dividing by n-1 rather than n (default: true).

# Returns
- `Vector{Float64}`: One downside deviation per column, in column order. A column with no
    returns below `mar` yields 0.0.

# Throws
- `AssertionError`: If `returns` holds non-numeric data.

# Notes
- The deviations are divided by the count of *below-MAR* observations, not by the total
    number of periods.
"""
function downside_deviation(returns::TimeArray, mar::Number=0; corrected::Bool=true)

    @assert all(eltype(values(returns)) <: Real) "All columns must contain numeric data"
    
    cols = colnames(returns)
    results = Vector{Float64}(undef, length(cols))
    
    for (idx, col) in enumerate(cols)
        ret_vals = filter(!isnan, values(returns[col]))
        negative_rets = ret_vals[ret_vals .< mar]
        
        if isempty(negative_rets)
            results[idx] = 0.0
        else
            squared_deviations = (negative_rets .- mar).^2
            n = length(negative_rets)
            # Use Bessel's correction if corrected is true and n > 1
            # Otherwise, use n for the denominator
            denominator = corrected && n > 1 ? n - 1 : n
            downside_var = sum(squared_deviations) / denominator
            results[idx] = sqrt(downside_var)
        end
    end
    
    return results
end


"""
    annualized_downside_deviation(returns::TimeArray, mar::Number = 0, periods_per_year::Int = 52;
                                  corrected::Bool = true) -> Vector{Float64}

Compute the annualized downside deviation of each asset.

The periodic figure from [`downside_deviation`](@ref) scaled by the square root of
`periods_per_year`.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `mar::Number`: Per-period minimum acceptable return (default: 0).
- `periods_per_year::Int`: Number of periods in a year (default: 52, i.e. weekly data).
- `corrected::Bool`: Apply Bessel's correction (default: true).

# Returns
- `Vector{Float64}`: One annualized downside deviation per column, in column order.

# Throws
- `AssertionError`: If `returns` has fewer than two data points, or `periods_per_year` is
    not positive.

# Notes
- `mar` is a *per-period* threshold. To use an annual target, divide it by `periods_per_year`
    before passing it in.
"""
function annualized_downside_deviation(returns::TimeArray, mar::Number=0, periods_per_year::Int=52; corrected::Bool=true)

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert periods_per_year > 0 "periods_per_year must be greater than 0"
    
    downs = downside_deviation(returns, mar, corrected = corrected)
    return downs .* sqrt(periods_per_year)
end


######################################## CAPM METRICS #############################################################
"""
    beta(returns::TimeArray, benchmark_returns::TimeArray) -> Vector{Float64}

Compute each asset's sensitivity to the benchmark.

    beta = cov(asset returns, benchmark returns) / var(benchmark returns)

A beta of 1 moves with the benchmark, above 1 amplifies it, below 1 damps it, and a negative
beta moves against it.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `benchmark_returns::TimeArray`: Single-column benchmark returns on the same timestamps.

# Returns
- `Vector{Float64}`: One beta per column, in column order. A benchmark with zero variance
    yields NaN.

# Throws
- `ErrorException`: If the timestamps of the two TimeArrays do not match.
- `AssertionError`: If either TimeArray contains NaN values. Drop or fill them first.
"""
function beta(returns::TimeArray, benchmark_returns::TimeArray)

    # Ensure both TimeArrays have the same timestamps
    if timestamp(returns) != timestamp(benchmark_returns)
        error("The timestamps of returns and benchmark_returns must match.")
    end

    ret_vals = values(returns)
    # vec() so cov() and var() below return scalars rather than 1x1 matrices.
    bench_vals = vec(values(benchmark_returns))

    # Assert no NaN values in returns
    @assert all(!isnan, ret_vals) "The returns TimeArray contains NaN values, which are not allowed."
    # Assert no NaN values in benchmark_returns
    @assert all(!isnan, bench_vals) "The benchmark_returns TimeArray contains NaN values, which are not allowed."

    n_cols = size(ret_vals, 2)
    betas = Vector{Float64}(undef, n_cols)
    var_bench = var(bench_vals)
    for col in 1:n_cols
        covar = cov(ret_vals[:, col], bench_vals)
        betas[col] = var_bench != 0 ? covar / var_bench : NaN  # Avoid division by zero
    end
    return betas

end


"""
    alpha(returns::TimeArray, benchmark_returns::TimeArray, risk_free_rate::Number = 0.0) -> Vector{Float64}

Compute each asset's return in excess of what its beta alone would predict.

    alpha = mean asset return - [risk-free rate + beta * (mean benchmark return - risk-free rate)]

Positive alpha is the part of performance not explained by simply taking on benchmark
exposure, which is the part attributable to skill or to the factor itself.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `benchmark_returns::TimeArray`: Single-column benchmark returns on the same timestamps.
- `risk_free_rate::Number`: Per-period risk-free rate (default: 0.0).

# Returns
- `Vector{Float64}`: One alpha per column, in column order, expressed per period in the same
    units as the input returns.

# Throws
- `ErrorException`: If the timestamps of the two TimeArrays do not match.
- `AssertionError`: If either TimeArray contains NaN values.

# Notes
- `risk_free_rate` is a *per-period* rate, matching the frequency of the returns.
"""
function alpha(returns::TimeArray, benchmark_returns::TimeArray, risk_free_rate::Number=0.0)

    # Ensure both TimeArrays have the same timestamps
    if timestamp(returns) != timestamp(benchmark_returns)
        error("The timestamps of returns and benchmark_returns must match.")
    end

    ret_vals = values(returns)
    bench_vals = values(benchmark_returns)
    # Assert no NaN values in returns
    @assert all(!isnan, ret_vals) "The returns TimeArray contains NaN values, which are not allowed."
    # Assert no NaN values in benchmark_returns
    @assert all(!isnan, bench_vals) "The benchmark_returns TimeArray contains NaN values, which are not allowed."

  
    betas = beta(returns, benchmark_returns)
    alphas = Vector{Float64}(undef, length(colnames(returns)))
    bk_mu = mean(vec(values(benchmark_returns)))
    actual_returns = column_mean(returns)

    # Calculate alpha for each column
    for (idx, col) in enumerate(colnames(returns))
        # Calculate expected return based on CAPM
        expected_return = risk_free_rate + betas[idx] * (bk_mu - risk_free_rate)
        # Calculate alpha
        alphas[idx] = actual_returns[idx] - expected_return
    end

    return alphas
end


################################## RISK-ADJUSTED RATIOS ###################################

"""
    arithmetic_sharpe_ratio(returns::TimeArray, risk_free_rate::Number = 0) -> Vector{Float64}

Compute the per-period Sharpe ratio of each asset, using the arithmetic mean return.

    Sharpe = (mean return - risk-free rate) / standard deviation

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `risk_free_rate::Number`: Per-period risk-free rate (default: 0).

# Returns
- `Vector{Float64}`: One Sharpe ratio per column, in column order, on a per-period basis.
    A column with zero volatility yields NaN.

# Throws
- `AssertionError`: If `returns` has fewer than two data points, or holds non-numeric data.

# Notes
- The result is per period, not annual. Use [`annualized_sharpe_ratio`](@ref) to compare
    against the figures usually quoted.
"""
function arithmetic_sharpe_ratio(returns::TimeArray, risk_free_rate::Number=0)

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert all(eltype(values(returns)) <: Real) "All columns must contain numeric data"

    vals = values(returns)
    excess_rets = column_mean(returns) .- risk_free_rate
    stdev_rets = stdev(returns)

    ar_sharpe_ratio = [stdev_rets[i] != 0 ? excess_rets[i] / stdev_rets[i] : NaN for i in 1:length(stdev_rets)]
    
    return ar_sharpe_ratio
end


"""
    geom_sharpe_ratio(returns::TimeArray, risk_free_rate::Number = 0) -> Vector{Float64}

Compute the per-period Sharpe ratio of each asset, using the compounded mean return.

    Sharpe = (geometric mean return - risk-free rate) / standard deviation

Identical to [`arithmetic_sharpe_ratio`](@ref) except that the numerator accounts for
compounding, which gives a slightly lower and more realistic figure for volatile series.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `risk_free_rate::Number`: Per-period risk-free rate (default: 0).

# Returns
- `Vector{Float64}`: One Sharpe ratio per column, in column order, on a per-period basis.
    A column with zero volatility yields NaN.

# Throws
- `AssertionError`: If `returns` has fewer than two data points, or holds non-numeric data.
"""
function geom_sharpe_ratio(returns::TimeArray, risk_free_rate::Number=0)

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert all(eltype(values(returns)) <: Real) "All columns must contain numeric data"

    vals = values(returns)
    excess_rets = geom_return(returns) .- risk_free_rate
    stdev_rets = stdev(returns)

    geom_sharpe_ratio = [stdev_rets[i] != 0 ? excess_rets[i] / stdev_rets[i] : NaN for i in 1:length(stdev_rets)]
    
    return geom_sharpe_ratio
end


"""
    annualized_sharpe_ratio(returns::TimeArray, periods_per_year::Int; risk_free_rate = 0.0) -> Vector{Float64}

Compute the annualized Sharpe ratio of each asset.

    Sharpe = (annualized return - risk-free rate) / annualized standard deviation

This is the conventionally quoted form, built from [`annualized_return`](@ref) and
[`annualized_stdev`](@ref) rather than by scaling a per-period ratio.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `periods_per_year::Int`: Number of periods in a year (252 daily, 52 weekly, 12 monthly).
- `risk_free_rate`: *Annual* risk-free rate, subtracted from the annualized return
    (default: 0.0).

# Returns
- `Vector{Float64}`: One annualized Sharpe ratio per column, in column order.

# Throws
- `AssertionError`: If `returns` has fewer than two data points, or `periods_per_year` is
    not positive.
"""
function annualized_sharpe_ratio(returns::TimeArray, periods_per_year::Int; risk_free_rate=0.0)

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert periods_per_year > 0 "periods_per_year must be greater than 0"

    annual_rets = annualized_return(returns, periods_per_year) .- risk_free_rate
    return annual_rets ./ annualized_stdev(returns, periods_per_year)
end


"""
    sortino_ratio(returns::TimeArray; risk_free_rate::Number = 0.0, mar::Number = 0.0,
                  corrected::Bool = true) -> Vector{Float64}

Compute the per-period Sortino ratio of each asset.

    Sortino = (mean return - risk-free rate) / downside deviation

The Sortino ratio is the Sharpe ratio with [`downside_deviation`](@ref) in the denominator
instead of total volatility, so an asset is not penalised for large gains.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `risk_free_rate::Number`: Per-period risk-free rate (default: 0.0).
- `mar::Number`: Per-period minimum acceptable return defining the downside (default: 0.0).
- `corrected::Bool`: Apply Bessel's correction in the downside deviation (default: true).

# Returns
- `Vector{Float64}`: One Sortino ratio per column, in column order, on a per-period basis.
    A column with no downside (zero downside deviation) yields NaN.

# Notes
- All arguments are keywords. Call it as `sortino_ratio(returns, mar = 0.01)`, not
    positionally.
"""
function sortino_ratio(returns::TimeArray; risk_free_rate::Number=0.0, mar::Number=0.0,
    corrected::Bool=true)

    # Calculate mean returns for each column
    excess_rets = column_mean(returns) .- risk_free_rate
    # Calculate downside deviation
    down_dev = downside_deviation(returns, mar; corrected = corrected)
    
    # Avoid division by zero
    sortino_ratios = [down_dev[i] != 0 ? excess_rets[i] / down_dev[i] : NaN for i in eachindex(down_dev)]
    return sortino_ratios
end


"""
    down_capture(returns::TimeArray, benchmark_returns::TimeArray, mar::Number = 0) -> Vector{Float64}

Compute how much of the benchmark's losses each asset absorbed.

Only the periods where the benchmark fell below `mar` are considered; the asset's mean return
over those periods is divided by the benchmark's. Lower is better: 0.8 means the asset lost
only 80% as much as the benchmark did in down markets, while above 1 means it lost more.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `benchmark_returns::TimeArray`: Single-column benchmark returns on the same timestamps.
- `mar::Number`: Threshold below which the benchmark counts as down (default: 0).

# Returns
- `Vector{Float64}`: One down capture ratio per column, in column order. All NaN if the
    benchmark never fell below `mar`, and NaN for a benchmark whose mean down return is zero.

# Throws
- `ErrorException`: If the timestamps of the two TimeArrays do not match.
"""
function down_capture(returns::TimeArray, benchmark_returns::TimeArray, mar::Number = 0)

    # Ensure both TimeArrays have the same timestamps
    if timestamp(returns) != timestamp(benchmark_returns)
        error("The timestamps of returns and benchmark_returns must match.")
    end

    # Identify periods where benchmark is underperforming (down market).
    # vec() keeps this working whether the benchmark holds a vector or a single-column matrix.
    down_market = findall(vec(values(benchmark_returns)) .< mar)

    # Check if there are any down markets to analyze
    if isempty(down_market)
        return fill(NaN, length(colnames(returns)))  # Return NaNs if no down capture can be calculated
    end

    # Filter returns and benchmark returns for down market periods
    portfolio_down = returns[down_market]
    benchmark_down = benchmark_returns[down_market]

    # Calculate down capture ratio for each column
    dc_ratio = Vector{Float64}(undef, length(colnames(returns)))
    avg_benchmark_down = column_mean(benchmark_down)[1]

    for (idx, col) in enumerate(colnames(returns))
        avg_portfolio_down = column_mean(portfolio_down[col])[1]

        if avg_benchmark_down == 0
            dc_ratio[idx] = NaN  # Avoid division by zero
        else
            dc_ratio[idx] = avg_portfolio_down / avg_benchmark_down
        end
    end

    return dc_ratio
end


"""
    up_capture(returns::TimeArray, benchmark_returns::TimeArray, mar::Number = 0) -> Vector{Float64}

Compute how much of the benchmark's gains each asset captured.

Only the periods where the benchmark rose above `mar` are considered; the asset's mean return
over those periods is divided by the benchmark's. Higher is better: 1.2 means the asset gained
20% more than the benchmark did in up markets.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `benchmark_returns::TimeArray`: Single-column benchmark returns on the same timestamps.
- `mar::Number`: Threshold above which the benchmark counts as up (default: 0).

# Returns
- `Vector{Float64}`: One up capture ratio per column, in column order. All NaN if the
    benchmark never rose above `mar`, and NaN for a benchmark whose mean up return is zero.

# Throws
- `ErrorException`: If the timestamps of the two TimeArrays do not match.
"""
function up_capture(returns::TimeArray, benchmark_returns::TimeArray, mar::Number = 0)

    # Ensure both TimeArrays have the same timestamps
    if timestamp(returns) != timestamp(benchmark_returns)
        error("The timestamps of returns and benchmark_returns must match.")
    end

    # Identify periods where the benchmark is outperforming (up market).
    # vec() keeps this working whether the benchmark holds a vector or a single-column matrix.
    up_market = findall(vec(values(benchmark_returns)) .> mar)

    # Check if there are any up markets to analyze
    if isempty(up_market)
        return fill(NaN, length(colnames(returns)))  # Return NaNs if no up capture can be calculated
    end

    # Filter returns and benchmark returns for up market periods
    portfolio_up = returns[up_market]
    benchmark_up = benchmark_returns[up_market]

    # Calculate down capture ratio for each column
    uc_ratio = Vector{Float64}(undef, length(colnames(returns)))
    avg_benchmark_up = column_mean(benchmark_up)[1]

    for (idx, col) in enumerate(colnames(returns))
        avg_portfolio_up = column_mean(portfolio_up[col])[1]

        if avg_benchmark_up == 0
            uc_ratio[idx] = NaN  # Avoid division by zero
        else
            uc_ratio[idx] = avg_portfolio_up / avg_benchmark_up
        end
    end

    return uc_ratio
end


"""
    overall_capture(returns::TimeArray, benchmark_returns::TimeArray, mar::Number = 0) -> Vector{Float64}

Compute the capture ratio: upside captured relative to downside suffered.

    overall capture = up capture / down capture

This condenses [`up_capture`](@ref) and [`down_capture`](@ref) into one number. Above 1 is
favourable, meaning the asset keeps more of the benchmark's gains than of its losses.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `benchmark_returns::TimeArray`: Single-column benchmark returns on the same timestamps.
- `mar::Number`: Threshold separating up from down markets (default: 0).

# Returns
- `Vector{Float64}`: One overall capture ratio per column, in column order.

# Notes
- The ratio is hard to read when down capture is negative, which happens when an asset gains
    on average while the benchmark falls. Inspect the two legs separately in that case.
"""
function overall_capture(returns::TimeArray, benchmark_returns::TimeArray, mar::Number = 0)

    dc = down_capture(returns, benchmark_returns, mar)
    uc = up_capture(returns, benchmark_returns, mar)
    oc_ratio = uc ./ dc

    return oc_ratio

end


"""
    tail_ratio(returns::TimeArray, benchmark_returns::TimeArray, percentile::Float64 = 0.95) -> Vector{Float64}

Compare how each asset behaves in the benchmark's best periods versus its worst.

    tail ratio = mean return in the benchmark's best tail / |mean return in its worst tail|

The tails are defined by the *benchmark*, not by the asset's own returns, so this measures
behaviour in extreme markets rather than the shape of the asset's own distribution. Above 1
means the asset gains more in the best periods than it loses in the worst.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `benchmark_returns::TimeArray`: Single-column benchmark returns on the same timestamps.
- `percentile::Float64`: Upper tail cutoff (default: 0.95). The lower tail is its complement,
    so the default compares the benchmark's best 5% of periods against its worst 5%.

# Returns
- `Vector{Float64}`: One tail ratio per column, in column order. All NaN if either tail is
    empty, and NaN for a column whose mean worst-tail return is zero.

# Throws
- `ErrorException`: If the timestamps of the two TimeArrays do not match.
"""
function tail_ratio(returns::TimeArray, benchmark_returns::TimeArray, percentile::Float64=0.95)

    # Step 1: Validate that timestamps match
    if timestamp(returns) != timestamp(benchmark_returns)
        error("The timestamps of returns and benchmark_returns must match.")
    end

    # Step 2: Extract benchmark returns values and compute percentiles.
    # vec() keeps this working whether the benchmark holds a vector or a single-column matrix.
    bench_vals = vec(values(benchmark_returns))
    lower_percentile = quantile(bench_vals, 1 - percentile)  # 5th percentile for worst tails
    upper_percentile = quantile(bench_vals, percentile)  # 95th percentile for best tails

    # Step 3: Identify tail periods
    worst_tail = findall(bench_vals .< lower_percentile)  # Worst-performing periods
    best_tail = findall(bench_vals .> upper_percentile)   # Best-performing periods

    # Step 4: Check if tail periods exist
    if isempty(worst_tail) || isempty(best_tail)
        return fill(NaN, length(colnames(returns)))  # Return NaN vector if no data
    end

    # Step 5: Filter portfolio returns for tail periods
    portfolio_worst = returns[worst_tail]
    portfolio_best = returns[best_tail]

    # Step 6: Compute tail ratios for each column
    tr_ratio = Vector{Float64}(undef, length(colnames(returns)))
    for (idx, col) in enumerate(colnames(returns))
        # Calculate average returns during worst and best tails, ignore NaN values (filter)
        avg_portfolio_worst = column_mean(portfolio_worst[col])[1]
        avg_portfolio_best = column_mean(portfolio_best[col])[1]

        # Compute tail ratio, handling division by zero
        if avg_portfolio_worst == 0
            tr_ratio[idx] = NaN
        else
            tr_ratio[idx] = avg_portfolio_best / abs(avg_portfolio_worst)
        end
    end

    # Step 7: Return the tail ratios
    return tr_ratio
end


"""
    treynor_ratio(returns::TimeArray, benchmark_returns::TimeArray, periods_per_year::Int,
                  risk_free_rate::Number = 0) -> Vector{Float64}

Compute each asset's excess return per unit of systematic risk.

    Treynor = (annualized return - risk-free rate) / beta

Where the Sharpe ratio divides by total volatility, the Treynor ratio divides by
[`beta`](@ref) alone. It therefore rewards only the market risk taken, which suits assets
held inside an already diversified portfolio.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `benchmark_returns::TimeArray`: Single-column benchmark returns on the same timestamps.
- `periods_per_year::Int`: Number of periods in a year (252 daily, 52 weekly, 12 monthly).
- `risk_free_rate::Number`: Annual risk-free rate (default: 0).

# Returns
- `Vector{Float64}`: One Treynor ratio per column, in column order. A column with zero beta
    yields NaN.

# Throws
- `ErrorException`: If the timestamps of the two TimeArrays do not match.
- `AssertionError`: If either TimeArray contains NaN values.

# Notes
- The ratio is meaningless for a near-zero or negative beta, where small changes in beta swing
    the result wildly.
"""
function treynor_ratio(returns::TimeArray, benchmark_returns::TimeArray, periods_per_year::Int, risk_free_rate::Number=0)

    # Assert no NaN values in returns or benchmark_returns
    @assert all(!isnan, values(returns)) "The returns TimeArray contains NaN values, which are not allowed."
    @assert all(!isnan, values(benchmark_returns)) "The benchmark_returns TimeArray contains NaN values, which are not allowed."

    if timestamp(returns) != timestamp(benchmark_returns)
        error("The timestamps of returns and benchmark_returns must match.")
    end

    betas = beta(returns, benchmark_returns)
    annual_rets = annualized_return(returns, periods_per_year) .- risk_free_rate

    treynor_ratios = [betas[i] != 0 ? annual_rets[i] / betas[i] : NaN for i in eachindex(betas)]
    return treynor_ratios

end


"""
    ulcer_index(returns::TimeArray) -> Vector{Float64}

Measure the depth *and* the duration of each asset's drawdowns.

    ulcer index = sqrt(mean(drawdown^2))

[`max_drawdown`](@ref) reports only the single worst moment. The ulcer index averages the
squared drawdown across every period, so an asset that spends a long time underwater scores
worse than one that recovers quickly from an equally deep fall. Lower is better.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.

# Returns
- `Vector{Float64}`: One ulcer index per column, in column order, as a positive decimal in
    the same units as the drawdowns.

# Throws
- `AssertionError`: If `returns` contains fewer than two data points.
"""
function ulcer_index(returns::TimeArray)

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    
    # Calculate the cumulative maximum of the returns
    dd = drawdowns(returns)
    
    # Calculate the Ulcer Index as the mean of the drawdown values
    ulcer_index = sqrt.(column_mean(dd.^2))
    
    return ulcer_index
end


"""
    ulcer_performance_index(returns::TimeArray, periods_per_year) -> Vector{Float64}

Compute return per unit of drawdown pain, also known as the Martin ratio.

    UPI = annualized return / ulcer index

A Sharpe-like ratio that swaps volatility for [`ulcer_index`](@ref), so it judges an asset by
how deep and how prolonged its drawdowns were rather than by how much it wobbled.

# Arguments
- `returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `periods_per_year`: Number of periods in a year (252 daily, 52 weekly, 12 monthly).

# Returns
- `Vector{Float64}`: One ulcer performance index per column, in column order. A column that
    never drew down (zero ulcer index) yields NaN.

# Throws
- `AssertionError`: If `returns` contains fewer than two data points.
"""
function ulcer_performance_index(returns::TimeArray, periods_per_year)

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    
    annual_rets = annualized_return(returns, periods_per_year)
    ulcer_idx = ulcer_index(returns)
    
    # Avoid division by zero
    upi = [ulcer_idx[i] != 0 ? annual_rets[i] / ulcer_idx[i] : NaN for i in eachindex(ulcer_idx)]
    
    return upi
end


################################### SUMMARY TABLES ###################################


"""
    performance_table(ta_returns::TimeArray, benchmark_returns::TimeArray, periods_per_year::Int,
                      mar::Number = 0) -> DataFrame

Assemble the full set of summary performance statistics for a group of assets.

This is the headline report: return, risk, benchmark-relative and drawdown-based measures
collected into one table, one column per asset. See [`quantile_performance_table`](@ref) for
the equivalent report across factor quantiles.

# Arguments
- `ta_returns::TimeArray`: Periodic returns as decimals, one column per asset.
- `benchmark_returns::TimeArray`: Single-column benchmark returns on the same timestamps.
- `periods_per_year::Int`: Number of periods in a year (252 daily, 52 weekly, 12 monthly).
- `mar::Number`: Threshold separating up from down markets, also used as the minimum
    acceptable return of the Sortino ratio (default: 0).

# Returns
- `DataFrame` with a leading `Stat` column and one column per asset. The rows are Annual
    Return, Annual StDev, Sharpe Ratio, Sortino Ratio, Max Drawdowns, Beta, Alpha, Treynor
    Ratio, Ulcer Ratio, Tail Ratio, Down Capture, Up Capture and Overall Capture.

# Throws
- `AssertionError`: If the asset and benchmark row counts do not match.
- `AssertionError`: If either TimeArray contains NaN values, which `beta` and `alpha` reject.

# Notes
- Every value is left on its natural scale: returns and drawdowns are decimals, not percent.
- Pass the result to [`plot_performance_table`](@ref) to chart the headline rows.
"""
function performance_table(ta_returns::TimeArray, benchmark_returns::TimeArray, periods_per_year::Int, mar::Number = 0)

    @assert size(ta_returns, 1) == size(benchmark_returns,1) "Asset returns and benchmark row counts do not match."

    asset_names = colnames(ta_returns)
    annual_returns = annualized_return(ta_returns, periods_per_year)
    annual_std = annualized_stdev(ta_returns, periods_per_year)
    sharpe_ratios = annualized_sharpe_ratio(ta_returns, periods_per_year)
    sortino_ratios = sortino_ratio(ta_returns, mar=mar)
    max_dds = max_drawdown(ta_returns)
    betas = beta(ta_returns, benchmark_returns)
    alphas = alpha(ta_returns, benchmark_returns) 
    treynor_ratios = treynor_ratio(ta_returns, benchmark_returns, periods_per_year)   
    ulcer_perf_index = ulcer_performance_index(ta_returns, periods_per_year)
    tail_ratios = tail_ratio(ta_returns, benchmark_returns)
    dc_ratios = down_capture(ta_returns, benchmark_returns, mar)
    uc_ratios = up_capture(ta_returns, benchmark_returns, mar)
    oc_ratios = overall_capture(ta_returns, benchmark_returns, mar)
 

    stats_names = ["Annual Return", "Annual StDev", "Sharpe Ratio", "Sortino Ratio", 
    "Max Drawdowns", "Beta", "Alpha","Treynor Ratio", "Ulcer Ratio", "Tail Ratio", 
    "Down Capture", "Up Capture", "Overall Capture"]

    summary_stats_table = DataFrame([annual_returns, annual_std, sharpe_ratios, sortino_ratios, 
    max_dds, betas, alphas, treynor_ratios, ulcer_perf_index, tail_ratios,
    dc_ratios, uc_ratios, oc_ratios], :auto)

    summary_stats_table = permutedims(summary_stats_table)
    summary_stats_table = DataFrames.rename(summary_stats_table, asset_names)


    summary_stats_table[!, "Stat"] = stats_names
    summary_stats_table = select(summary_stats_table, :Stat, asset_names...)

    #summary_stats_table = transform(ta_returns, names(ta_returns, AbstractFloat) .=> ByRow(x -> round(x, digits=6)) .=> names(ta_returns, AbstractFloat))

    return summary_stats_table

end