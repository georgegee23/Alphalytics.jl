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





function to_pctchange(prices::TimeArray, window::Int64=1)

    """

    Calculate the percentage change of prices over a specified window.

    This function computes the percentage change of prices in a TimeArray over a given window size.
    It handles missing values by replacing them with NaN and uses padding to maintain the original
    time series length.

    # Arguments
    - `prices::TimeArray`: A TimeArray containing price data. Can be single or multi-column.
    - `window::Int64`: The size of the window over which to calculate the percentage change.
        Must be at least 1.

    # Returns
    - `TimeArray`: A new TimeArray with the same timestamps and column names as the input,
        but with values representing the percentage changes.

    # Details
    - Missing values in the input are replaced with NaN.
    - The function uses `TimeSeries.lag` with padding to create a lagged version of the prices.
    - Percentage change is calculated as (current_price / lagged_price) - 1.
    - The first `window` number of rows in the result will contain NaN values due to insufficient
        historical data for calculation.

    """

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

function to_cumulative_growth(returns::TimeArray, init_value::Number = 1)

    """
    Compute the cumulative product of returns, preserving NaN locations.

    This approach preserves the location of NaN values in the output.
    The cumulative product continues from the last non-NaN value after encountering a NaN.
    It treats the input as returns, so it adds 1 to each value before multiplying (assuming returns are expressed as percentages or decimals).

    Parameters:
    - returns: TimeArray containing return data with possible NaN values

    Returns:
    - TimeArray with cumulative products, preserving NaN locations
    """

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

function to_cummax(returns::TimeArray)

    """
    cumulative_max(returns::TimeArray)

    Compute the cumulative maximum (running maximum) of a `TimeArray` object, handling both 
    single-column and multi-column data. For each timestamp, the cumulative maximum is the 
    highest value observed up to that point in the series, with `NaN` values preserved where 
    the input is `NaN`. The function supports time series with initial `NaN` values, starting 
    the maximum calculation from the first non-`NaN` entry per column.

    # Arguments
    - `ta::TimeArray`: A `TimeArray` object containing price data. The `values` field can 
    be a `Vector` (single series) or a `Matrix` (multiple series), typically of type `Float64`.

    # Returns
    - A new `TimeArray` with the same timestamps and column names, containing the 
    cumulative maximum values. The output matches the shape of `ta`, with `NaN` 
    where the input is `NaN` and the running maximum elsewhere.
    """

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


function arithmetic_return(returns::TimeArray)
    """
    Compute the mean return of each column in a TimeArray of returns.

    This function calculates the arithmetic mean of returns for each column, ignoring NaN values. 
    If a column has fewer than 2 non-NaN values, its mean is set to NaN.

    Parameters:
    - returns: TimeArray containing periodic asset returns

    Returns:
    - An array of mean returns, one for each column.
    """
   
    return column_mean(returns)
end


function geom_return(returns::TimeArray)
    """
    Compute the geometric mean return of each column in a TimeArray of returns.

    This function calculates the geometric mean of returns for each column, ignoring NaN values. 
    If a column has fewer than 2 non-NaN values, its geometric mean is set to NaN.

    Parameters:
    - returns: TimeArray containing periodic asset returns

    Returns:
    - An array of geometric mean returns, one for each column.
    """
    
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


function annualized_return(returns::TimeArray, periods_per_year::Int)

    """
    Compute annualized return of each column in a TimeArray of returns.

    Note: This function computes the annualized return even if columns data have different starts. 
    To make time equivalent comparisons, drop NaN values before using. 

    # Arguments
    - `returns::TimeArray`: A TimeArray containing periodic return data (e.g., daily or monthly returns).
    - `periods_per_year::Int`: The number of periods in a year (e.g., 252 for daily, 12 for monthly).

    # Returns
    - A vector of annualized returns, one for each column in the input TimeArray.
    """

    @assert length(returns) >= 2 "TimeArray must contain at least two data points"

    compounded_growth = to_cumulative_growth(returns)
    n_periods = size(compounded_growth, 1)
    ann_rets = (last.(eachcol(values(compounded_growth))) .^ (periods_per_year / n_periods)) .- 1

    return ann_rets

end


function drawdowns(returns::TimeArray)

    """
    Calculate the drawdown of a TimeArray of returns.

    This function computes the drawdown for each column in a TimeArray, which is defined as the 
    difference between the cumulative maximum and the current value. The drawdown is expressed 
    as a percentage of the cumulative maximum.

    # Arguments
    - `returns`: A TimeArray object containing return data.

    # Returns
    - A new TimeArray with the same timestamps and column names, containing the drawdown values.
    """
    cumgrowth = to_cumulative_growth(returns)
    # Calculate the cumulative maximum of the returns
    dd = cumgrowth ./ to_cummax(returns) .-1
    dd = TimeSeries.rename!(dd, colnames(returns))

    return dd
end


function max_drawdown(returns::TimeArray)
    """
    Calculate the maximum drawdown of a TimeArray of returns.

    This function computes the maximum drawdown for each column in a TimeArray, which is defined as the 
    maximum observed drawdown from the cumulative maximum.

    # Arguments
    - `returns`: A TimeArray object containing return data.

    # Returns
    - A DataFrame with two columns: "Asset" and "MaxDD", where "Asset" contains the names of the 
      assets and "MaxDD" contains the corresponding maximum drawdown values.
    """
    
    dd = drawdowns(returns)
    abs_dd = abs.(dd)
    vals = values(abs_dd)
    max_vals = [maximum(filter(!isnan, vals[:, col])) for col in 1:size(vals, 2)]
    return max_vals
end



#################################### VOLATILITY STATS ########################################

function stdev(returns::TimeArray)
    """
    Compute the standard deviation of each column in a TimeArray of returns.

    This function calculates the sample standard deviation for each column, ignoring NaN values. 
    If a column has fewer than 2 non-NaN values, its standard deviation is set to NaN.

    Parameters:
    - returns: TimeArray containing periodic asset returns

    Returns:
    - An array of standard deviations, one for each column.
    """
    
    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert all(eltype(values(returns)) <: Real) "All columns must contain numeric data"
    
    vals = values(returns)
    std_dev = [length(filter(!isnan, vals[:, col])) >= 2 ? std(filter(!isnan, vals[:, col])) : NaN for col in 1:size(vals, 2)]
    return std_dev
end


function annualized_stdev(returns::TimeArray, periods_per_year::Int)
 
    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    
    std_dev = stdev(returns)
    # Convert to a vector for easier manipulation
    annual_std_dev = std_dev .* sqrt(periods_per_year)
    return annual_std_dev
end


function downside_deviation(returns::TimeArray, mar::Number=0; corrected::Bool=true)
    """
    Calculate the downside deviation of returns in a TimeArray.

    Parameters:
    - returns: TimeArray containing asset returns
    - mar: Minimum acceptable return (MAR)
    - corrected: Boolean flag indicating whether to use Bessel's correction (default: true)

    Returns:
    - An array of downside deviations.
    """
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


function annualized_downside_deviation(returns::TimeArray, mar::Number=0, periods_per_year::Int=52; corrected::Bool=true)
    """
    Calculate the annualized downside deviation of returns in a TimeArray.

    Parameters:
    - returns: TimeArray containing periodic asset returns
    - mar: Per-period minimum acceptable return (MAR)
    - periods_per_year: Number of periods in a year (default: 52 for weekly data)
    - corrected: Boolean flag indicating whether to use Bessel's correction (default: true)

    Returns:
    - An array of annualized downside deviations.
    """
    
    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert periods_per_year > 0 "periods_per_year must be greater than 0"
    
    downs = downside_deviation(returns, mar, corrected = corrected)
    return downs .* sqrt(periods_per_year)
end


######################################## CAPM METRICS #############################################################
function beta(returns::TimeArray, benchmark_returns::TimeArray)

    """
    Compute the beta of each column in a TimeArray of returns.

    Beta is a measure of the sensitivity of an asset's returns to the returns of a benchmark. 
    It is calculated as the covariance of the asset's returns with the benchmark's returns, 
    divided by the variance of the benchmark's returns.

    Beta (β) = Covariance (Asset Returns, Market Returns) / Variance (Market Returns) 

    Parameters:
    - returns: TimeArray containing portfolio returns
    - benchmark_returns: TimeArray containing benchmark returns

    Returns:
    - Vector of betas for each column in the TimeArray
    """

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

    n_cols = size(ret_vals, 2)
    betas = Vector{Float64}(undef, n_cols)
    for col in 1:n_cols
        # Calculate covariance and variance
        covar = cov(ret_vals[:, col], bench_vals)
        var_bench = var(bench_vals)
        # Calculate beta
        betas[col] = var_bench != 0 ? covar / var_bench : NaN  # Avoid division by zero
    end
    return betas 

end


function alpha(returns::TimeArray, benchmark_returns::TimeArray, risk_free_rate::Number=0.0)

    """
    Compute the alpha of each column in a TimeArray of returns.

    Alpha is a measure of the active return on an investment compared to a benchmark index. 
    It is calculated as the difference between the actual return and the expected return based on beta.

    Alpha (α) = Asset Return - [Risk-Free Rate + Beta * (Benchmark Return - Risk-Free Rate)]

    Parameters:
    - returns: TimeArray containing portfolio returns
    - benchmark_returns: TimeArray containing benchmark returns
    - risk_free_rate: Risk-free rate to subtract from expected returns (default: 0.0)

    Returns:
    - Vector of alphas for each column in the TimeArray
    """

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

    # Calculate alpha for each column
    for (idx, col) in enumerate(colnames(returns))
        # Calculate expected return based on CAPM
        expected_return = risk_free_rate + betas[idx] * (bk_mu - risk_free_rate)
        # Calculate actual return
        actual_return = column_mean(returns)[idx]
        # Calculate alpha
        alphas[idx] = actual_return - expected_return
    end

    return alphas
end


################################## RISK-ADJUSTED RATIOS ###################################

function arithmetic_sharpe_ratio(returns::TimeArray, risk_free_rate::Number=0)
    """
    Compute the arithmetic Sharpe ratio of each column in a TimeArray of returns.

    The Sharpe ratio is calculated as (Mean Return - Risk-Free Rate) / Standard Deviation, 
    where both mean return and standard deviation are computed per period, ignoring NaN values.

    Parameters:
    - returns: TimeArray containing periodic asset returns
    - risk_free_rate: Per-period risk-free rate (default: 0)

    Returns:
    - An array of arithmetic Sharpe ratios, one for each column.
    """
    
    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert all(eltype(values(returns)) <: Real) "All columns must contain numeric data"

    vals = values(returns)
    excess_rets = column_mean(returns) .- risk_free_rate
    stdev_rets = stdev(returns)

    ar_sharpe_ratio = [stdev_rets[i] != 0 ? excess_rets[i] / stdev_rets[i] : NaN for i in 1:length(stdev_rets)]
    
    return ar_sharpe_ratio
end


function geom_sharpe_ratio(returns::TimeArray, risk_free_rate::Number=0)
    """
    Compute the geometric Sharpe ratio of each column in a TimeArray of returns.

    The Sharpe ratio is calculated as (Geometric Mean Return - Risk-Free Rate) / Standard Deviation, 
    where geometric mean return and standard deviation are computed per period, ignoring NaN values.

    Parameters:
    - returns: TimeArray containing periodic asset returns
    - risk_free_rate: Per-period risk-free rate (default: 0)

    Returns:
    - An array of geometric Sharpe ratios, one for each column.
    """
    
    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert all(eltype(values(returns)) <: Real) "All columns must contain numeric data"

    vals = values(returns)
    excess_rets = geom_return(returns) .- risk_free_rate
    stdev_rets = stdev(returns)

    geom_sharpe_ratio = [stdev_rets[i] != 0 ? excess_rets[i] / stdev_rets[i] : NaN for i in 1:length(stdev_rets)]
    
    return geom_sharpe_ratio
end


function annualized_sharpe_ratio(returns::TimeArray, periods_per_year::Int; risk_free_rate=0.0)

    """
    Compute annual sharpe ratio of each column in a TimeArray of returns.
    """
    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    @assert periods_per_year > 0 "periods_per_year must be greater than 0"

    annual_rets = annualized_return(returns, periods_per_year) .- risk_free_rate
    return annual_rets ./ annualized_stdev(returns, periods_per_year)
end


function sortino_ratio(returns::TimeArray; risk_free_rate::Number=0.0, mar::Number=0.0,
    corrected::Bool=true)
    """
    Compute Sortino ratio of each column in a TimeArray of returns.

    Sortino Ratio = (Mean Return - Risk-Free Rate) / Downside Deviation

    Parameters:
    - returns: TimeArray containing asset returns
    - mar: Minimum acceptable return (MAR) for downside deviation
    - risk_free_rate: Risk-free rate to subtract from mean returns (default: 0.0)
    - corrected: Boolean flag indicating whether to use Bessel's correction (default: true)

    Returns:
    - An array of Sortino ratios for each column.
    """
    
    # Calculate mean returns for each column
    excess_rets = column_mean(returns) .- risk_free_rate
    # Calculate downside deviation
    down_dev = downside_deviation(returns, mar; corrected = corrected)
    
    # Avoid division by zero
    sortino_ratios = [down_dev[i] != 0 ? excess_rets[i] / down_dev[i] : NaN for i in eachindex(down_dev)]
    return sortino_ratios
end


function down_capture(returns::TimeArray, benchmark_returns::TimeArray, mar::Number = 0)
    
    """
    Compute down capture for each column in a TimeArray of returns.

    Parameters:
    - returns: TimeArray containing portfolio returns
    - benchmark_returns: TimeArray containing benchmark returns
    - thresh_value: Threshold value to determine down markets (default: 0)

    Returns:
    - Vector of down capture ratios for each column in the TimeArray
    """

    # Ensure both TimeArrays have the same timestamps
    if timestamp(returns) != timestamp(benchmark_returns)
        error("The timestamps of returns and benchmark_returns must match.")
    end

    # Identify periods where benchmark is underperforming (down market)
    down_market = values(benchmark_returns) .< mar

    # Filter returns and benchmark returns for down market periods
    portfolio_down = returns[down_market]
    benchmark_down = benchmark_returns[down_market]  # Assuming benchmark data is in a column named 'Benchmark'

    # Check if there are any down markets to analyze
    if isempty(portfolio_down) || isempty(benchmark_down)
        return fill(NaN, length(colnames(returns)))  # Return NaNs if no down capture can be calculated
    end

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


function up_capture(returns::TimeArray, benchmark_returns::TimeArray, mar::Number = 0)
    
    """
    Compute down capture for each column in a TimeArray of returns.

    Parameters:
    - returns: TimeArray containing portfolio returns
    - benchmark_returns: TimeArray containing benchmark returns
    - thresh_value: Threshold value to determine down markets (default: 0)

    Returns:
    - Vector of down capture ratios for each column in the TimeArray
    """

    # Ensure both TimeArrays have the same timestamps
    if timestamp(returns) != timestamp(benchmark_returns)
        error("The timestamps of returns and benchmark_returns must match.")
    end

    # Identify periods where benchmark is underperforming (down market)
    up_market = values(benchmark_returns) .> mar

    # Filter returns and benchmark returns for down market periods
    portfolio_up = returns[up_market]
    benchmark_up = benchmark_returns[up_market]  # Assuming benchmark data is in a column named 'Benchmark'

    # Check if there are any down markets to analyze
    if isempty(portfolio_up) || isempty(benchmark_up)
        return fill(NaN, length(colnames(returns)))  # Return NaNs if no down capture can be calculated
    end

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


function overall_capture(returns::TimeArray, benchmark_returns::TimeArray, mar::Number = 0)

    """
    Compute overall capture for each column in a TimeArray of returns.

    Parameters:
    - returns: TimeArray containing portfolio returns
    - benchmark_returns: TimeArray containing benchmark returns
    - thresh_value: Threshold value to determine up markets (default: 0)

    Returns:
    - Vector of overall capture ratios for each column in the TimeArray
    """

    dc = down_capture(returns, benchmark_returns, mar)
    uc = up_capture(returns, benchmark_returns, mar)
    oc_ratio = uc ./ dc

    return oc_ratio

end


function tail_ratio(returns::TimeArray, benchmark_returns::TimeArray, percentile::Float64=0.95)
    """
    Compute the tail ratio for each column in a TimeArray of returns.

    The tail ratio is a risk-adjusted performance measure that assesses an investment's downside risk by comparing the average return during the best-performing tails (e.g., 95%) to the average return during the worst-performing tails (e.g., 5%).

    Tail Ratio = 95th Percentile Return / |5th Percentile Return|

    Parameters:
    - returns: TimeArray containing portfolio returns
    - benchmark_returns: TimeArray containing benchmark returns
    - percentile: Threshold value to determine tail markets (default: 0.0, though percentiles are used)

    Returns:
    - Vector of tail ratios for each column in the returns TimeArray
    """

    # Step 1: Validate that timestamps match
    if timestamp(returns) != timestamp(benchmark_returns)
        error("The timestamps of returns and benchmark_returns must match.")
    end

    # Step 2: Extract benchmark returns values and compute percentiles
    bench_vals = values(benchmark_returns)
    lower_percentile = quantile(bench_vals, 1 - percentile)  # 5th percentile for worst tails
    upper_percentile = quantile(bench_vals, percentile)  # 95th percentile for best tails

    # Step 3: Identify tail periods
    worst_tail = bench_vals .< lower_percentile  # Boolean mask for worst-performing periods
    best_tail = bench_vals .> upper_percentile   # Boolean mask for best-performing periods

    # Step 4: Filter portfolio returns for tail periods
    portfolio_worst = returns[worst_tail]
    portfolio_best = returns[best_tail]

    # Step 5: Check if tail periods exist
    if isempty(portfolio_worst) || isempty(portfolio_best)
        return fill(NaN, length(colnames(returns)))  # Return NaN vector if no data
    end

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


function treynor_ratio(returns::TimeArray, benchmark_returns::TimeArray, periods_per_year::Int, risk_free_rate::Number=0)

    """
    Compute the Treynor ratio for each column in a TimeArray of returns.

    The Treynor ratio measures excess return per unit of systematic risk (beta).
    Treynor Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Beta

    Parameters:
    - returns: TimeArray containing portfolio returns
    - benchmark_returns: TimeArray containing benchmark returns (single column assumed)
    - risk_free_rate: Risk-free rate (default: 0)

    Returns:
    - Vector of Treynor ratios for each column in the TimeArray
    """

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


function ulcer_index(returns::TimeArray)

    """
    Compute the Ulcer Index for each column in a TimeArray of returns.

    The Ulcer Index is a measure of downside risk that quantifies the depth and duration of drawdowns.

    Ulcer Index = Square Root of (Average of Squared Percentage Drawdowns)

    Parameters:
    - returns: TimeArray containing periodic asset returns

    Returns:
    - An array of Ulcer Index values, one for each column.
    """
    
    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    
    # Calculate the cumulative maximum of the returns
    dd = drawdowns(returns)
    
    # Calculate the Ulcer Index as the mean of the drawdown values
    ulcer_index = sqrt.(column_mean(dd.^2))
    
    return ulcer_index
end


function ulcer_performance_index(returns::TimeArray, periods_per_year)

    """
    Compute the Ulcer Performance Index for each column in a TimeArray of returns.

    The Ulcer Performance Index is a risk-adjusted performance measure that considers both the 
    average return and the Ulcer Index (a measure of downside risk).

    Ulcer Performance Index = Mean Return / Ulcer Index

    Parameters:
    - returns: TimeArray containing periodic asset returns

    Returns:
    - An array of Ulcer Performance Index values, one for each column.
    """
    
    @assert length(returns) >= 2 "TimeArray must contain at least two data points"
    
    annual_rets = annualized_return(returns, periods_per_year)
    ulcer_idx = ulcer_index(returns)
    
    # Avoid division by zero
    upi = [ulcer_idx[i] != 0 ? annual_rets[i] / ulcer_idx[i] : NaN for i in eachindex(ulcer_idx)]
    
    return upi
end


################################### SUMMARY TABLES ###################################


function performance_table(ta_returns::TimeArray, benchmark_returns::TimeArray, periods_per_year::Int, mar::Number = 0)

    """
    Compute table with summary performance statistics for asset returns.

    Parameters:
    - ta_returns: TimeArray containing returns for each asset
    - benchmark_returns: TimeArray containing benchmark returns
    - thresh_value: Threshold value for down markets (default: 0)
    - periods_per_year: Number of periods in a year 

    Returns:
    - DataFrame with performance metrics for each asset
    """   

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