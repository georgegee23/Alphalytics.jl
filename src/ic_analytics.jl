########################################## Information Coefficient Analysis (Spearman Rank Correlation Analysis) ###########################################################


####### CORRELATION ANALYSIS ##########################################################

"""
    rows_spearmanr(factors::TimeArray, returns::TimeArray) -> TimeArray

Compute the cross-sectional (row-wise) Spearman rank correlation between factors and returns.

This is the rank information coefficient (IC): for each timestamp, the securities are ranked
by factor score and by realised return, and the two rankings are correlated. Being rank based,
it measures monotonic association and is insensitive to outliers.

# Arguments
- `factors::TimeArray`: Factor scores, one column per security.
- `returns::TimeArray`: Realised returns, same dimensions and timestamps as `factors`.

# Returns
- `TimeArray`: A single column named `:SpearmanRank_IC` holding one correlation per timestamp.
    A row is NaN when fewer than two securities have both a factor score and a return.

# Throws
- `AssertionError`: If `factors` and `returns` do not have the same dimensions.

# Example
```julia
ic = rows_spearmanr(factor_scores, forward_returns)
mean(filter(!isnan, values(ic)))  # average IC over the sample
```
"""
function rows_spearmanr(factors::TimeArray, returns::TimeArray)

    factors_dims = size(factors)
    @assert factors_dims == size(returns) "TimeArrays must have the same dimensions"

    factors_mtx = values(factors)
    returns_mtx = values(returns)

    n_rows = factors_dims[1]
    correlations = Vector{Float64}(undef, n_rows)

    for idx in 1:n_rows
        x = @view factors_mtx[idx, :]
        y = @view returns_mtx[idx, :]

        valid_indices = .!isnan.(x) .& .!isnan.(y)
        x_clean = x[valid_indices]
        y_clean = y[valid_indices]

        if length(x_clean) > 1 && length(y_clean) > 1
            correlations[idx] = corspearman(x_clean, y_clean)
        else
            correlations[idx] = NaN
        end
    end

    ta_correlations = TimeArray(timestamp(factors), correlations, [:SpearmanRank_IC])

    return ta_correlations
end



"""
    rows_pearsonr(factors::TimeArray, returns::TimeArray) -> TimeArray

Compute the cross-sectional (row-wise) Pearson correlation between factors and returns.

This is the linear counterpart of [`rows_spearmanr`](@ref). It measures the strength of the
linear relationship between factor score and realised return, so unlike the rank IC it is
sensitive to outliers and to the scale of the factor.

# Arguments
- `factors::TimeArray`: Factor scores, one column per security.
- `returns::TimeArray`: Realised returns, same dimensions and timestamps as `factors`.

# Returns
- `TimeArray`: A single column named `:Pearson_IC` holding one correlation per timestamp.
    A row is NaN when fewer than two securities have both a factor score and a return.

# Throws
- `AssertionError`: If `factors` and `returns` do not have the same dimensions.
"""
function rows_pearsonr(factors::TimeArray, returns::TimeArray)

    factors_dims = size(factors)
    @assert factors_dims == size(returns) "TimeArrays must have the same dimensions"

    factors_mtx = values(factors)
    returns_mtx = values(returns)

    n_rows = factors_dims[1]
    correlations = Vector{Float64}(undef, n_rows)

    for idx in 1:n_rows
        x = @view factors_mtx[idx, :]
        y = @view returns_mtx[idx, :]

        valid_indices = .!isnan.(x) .& .!isnan.(y)
        x_clean = x[valid_indices]
        y_clean = y[valid_indices]

        if length(x_clean) > 1 && length(y_clean) > 1
            correlations[idx] = cor(x_clean, y_clean)
        else
            correlations[idx] = NaN
        end
    end

    ta_correlations = TimeArray(timestamp(factors), correlations, [:Pearson_IC])

    return ta_correlations
end


"""
    spearman_factor_decay(factors::TimeArray, returns::TimeArray, max_lags::Int) -> Vector{Pair{Int, Float64}}

Measure how quickly a factor's predictive power decays as it ages.

For each lag from 1 to `max_lags` the factor is shifted forward in time, re-aligned with the
returns it would then be predicting, and the mean cross-sectional rank IC is taken. A series
that holds its IC over many lags is slow to decay and tolerates less frequent rebalancing.

# Arguments
- `factors::TimeArray`: Factor scores, one column per security.
- `returns::TimeArray`: Realised returns, same dimensions and timestamps as `factors`.
- `max_lags::Int`: Largest lag to evaluate.

# Returns
- `Vector{Pair{Int, Float64}}`: One `lag => mean IC` pair per lag, ordered from lag 1 to
    `max_lags`.

# Throws
- `AssertionError`: If `factors` and `returns` do not have the same dimensions.

# Example
```julia
decay = spearman_factor_decay(factor_scores, forward_returns, 12)
last.(decay)  # the mean IC at each lag, as a plain vector
```
"""
function spearman_factor_decay(factors::TimeArray, returns:: TimeArray, max_lags::Int)

    @assert size(factors) == size(returns) "Factors and returns must have the same dimensions"
    
    factor_decay_corrs = Dict{Int, Float64}()

    for lag_value in 1:max_lags
        # Lag the factors
        factors_lagged = lag(factors, lag_value)
        
        # Align the returns, we drop the first 'lag_value' entries
        returns_aligned = returns[1+lag_value:end]
        
        # Compute Spearman correlation for each row
        spearman = rows_spearmanr(factors_lagged, returns_aligned)
        
        # Store the mean correlation for this lag in the dictionary
        factor_decay_corrs[lag_value] = mean(values(spearman))
    end

    # collect() first: a Dict has no order of its own, so sort its pairs by lag.
    return sort(collect(factor_decay_corrs), by = first)
end


"""
    mean_autocor(factors::TimeArray, maxlag::Int) -> Vector{Float64}

Compute the average autocorrelation of factor scores across securities.

Each security's own score history is autocorrelated, then the results are averaged across
securities lag by lag. High autocorrelation means scores are persistent, which implies low
turnover; low autocorrelation means the factor reshuffles its rankings quickly.

# Arguments
- `factors::TimeArray`: Factor scores, one column per security.
- `maxlag::Int`: Maximum lag for which to compute the autocorrelation.

# Returns
- `Vector{Float64}`: Mean autocorrelation per lag, of length `maxlag + 1`. The first entry
    is lag 0 and is therefore always 1.0; lag `k` sits at index `k + 1`.
- If no security has more than `maxlag` non-NaN scores, a warning is issued and a vector of
    `maxlag` NaNs is returned.

# Details
- NaN scores are dropped per security before the autocorrelation is computed, and securities
    with `maxlag` or fewer remaining observations are excluded entirely.
"""
function mean_autocor(factors::TimeArray, maxlag::Int)

    data = values(factors) |> eachcol .|> col -> filter(!isnan, col)
    clean_data = filter(v -> length(v) > maxlag, data)

    if isempty(clean_data)
        @warn "No factors with sufficient non-NaN data points"
        return fill(NaN, maxlag)
    end

    autocors_vecs = [autocor(x, 0:maxlag; demean=true) for x in clean_data]
    autocors_matrix = hcat(autocors_vecs...)
    mean_vec = autocors_matrix |> eachrow |> x -> filter.(!isnan, x) .|> mean 

    return mean_vec
end

"""
    rolling_mean_autocor(factors::TimeArray, acf_maxlag::Int, window::Int) -> TimeArray

Track how factor persistence changes over time by computing [`mean_autocor`](@ref) on a
rolling window.

# Arguments
- `factors::TimeArray`: Factor scores, one column per security.
- `acf_maxlag::Int`: The maximum lag for which to compute autocorrelations.
- `window::Int`: Number of observations in each rolling window.

# Returns
- `TimeArray`: One column per lag, named `ACF1` through `ACF<acf_maxlag>`, and one row per
    window. Lag 0 is dropped, since it is always 1.0. Each row is stamped with the timestamp
    of the last observation in its window, so the series starts at `timestamp(factors)[window]`.

# Throws
- `ErrorException`: If `factors` has fewer than `window + acf_maxlag` observations.

# Details
- A window whose autocorrelation cannot be computed is warned about and left uninitialised
    rather than aborting the whole series.
"""
function rolling_mean_autocor(factors::TimeArray, acf_maxlag::Int, window::Int)

    if size(factors, 1) < window + acf_maxlag
        error("TimeArray must have at least `window` + `acf_maxlag` observations.")
    end

    n_periods = size(factors, 1)
    n_windows = n_periods - window + 1
    
    # Pre-allocate matrix for results
    rolling_acfs = Matrix{Union{Float64}}(undef, n_windows, acf_maxlag)

    for start_idx in 1:n_windows
        end_idx = start_idx + window - 1
        w_factors = factors[start_idx:end_idx]
        
        try
            acfs = mean_autocor(w_factors, acf_maxlag)
            rolling_acfs[start_idx, :] = acfs[2:end]  # Exclude lag 0
        catch e
            @warn "Error calculating ACF for window starting at index $start_idx: $e"
        end
    end

    # Create TimeArray with results
    result_dates = timestamp(factors)[window:end]
    column_names = [Symbol("ACF$i") for i in 1:acf_maxlag]
    
    return TimeArray(result_dates, rolling_acfs, column_names)
end