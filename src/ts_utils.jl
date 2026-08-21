

####### TIME ARRAY MANIPULATIONS ##########################################################

"""
    row_mean(ta::TimeArray) -> TimeArray

Calculate the mean of each row in a `TimeArray`, ignoring NaN values.

# Arguments
- `ta::TimeArray`: Input TimeArray containing data with possible NaN values.

# Returns
- `TimeArray`: A new TimeArray with the same timestamps as the input, where each value represents 
  the mean of non-NaN values in the corresponding row of the input.

# Example
```julia
ta = TimeArray([1.0 2.0 NaN; 4.0 NaN 6.0], timestamp=1:2, colnames=[:A, :B, :C])
row_mean(ta)
```
"""
function row_mean(ta::TimeArray)
    ta_mtx = values(ta)
    # Calculate row means, ignoring NaN values
    row_mu = [mean(filter(!isnan, row)) for row in eachrow(ta_mtx)]
    # Create a meaningful column name
    col_name = [:Mean]
    return TimeArray(timestamp(ta), row_mu, col_name)
end


"""
    column_mean(ta::TimeArray) -> Vector{Float64}

Calculate the mean of each column in a `TimeArray`, ignoring NaN values.

# Arguments
- `ta::TimeArray`: Input TimeArray containing data with possible NaN values.

# Returns
- `Vector{Float64}`: One mean per column, in column order. This is the same shape as the
  other column statistics such as `stdev` and `downside_deviation`. Pair it with
  `colnames(ta)` to label the results.

# Example
```julia
ta = TimeArray([1.0 2.0 NaN; 4.0 NaN 6.0], timestamp=1:2, colnames=[:A, :B, :C])
column_mean(ta)
```
"""
function column_mean(ta::TimeArray)
    ta_mtx = values(ta)
    # Calculate column means, ignoring NaN values
    col_mu = [mean(filter(!isnan, col)) for col in eachcol(ta_mtx)]
    return col_mu
end


"""
    rowwise_ordinalrank(ta::TimeArray) -> TimeArray

Compute ordinal ranks for each value in the `TimeArray` along rows, ignoring `NaN`s.

For each row, non-NaN values are ranked in ascending order (smallest gets rank 1). NaN positions remain NaN in the output.

# Arguments
- `ta::TimeArray`: A `TimeArray` containing numerical data.

# Returns
- `TimeArray`: A new TimeArray with the same timestamps and column names as the input, but with values replaced by their row-wise ordinal ranks. NaN values are preserved in their original positions.

# Example
```julia
ta = TimeArray([1.0 3.0 NaN; 2.0 1.0 4.0], timestamp=1:2, colnames=[:A, :B, :C])
rowwise_ordinalrank(ta)
```
"""
function rowwise_ordinalrank(ta::TimeArray)
    values_matrix = values(ta)
    rank_matrix = similar(values_matrix, Float64)
    for i in axes(values_matrix, 1)
        row = values_matrix[i, :]
        non_nan_indices = findall(!isnan, row)
        non_nan_values = row[non_nan_indices]
        if !isempty(non_nan_values)
            row_ranks = ordinalrank(non_nan_values)
            rank_matrix[i, non_nan_indices] .= row_ranks
        end
        # Preserve NaN values in rank matrix
        rank_matrix[i, setdiff(1:length(row), non_nan_indices)] .= NaN
    end
    return TimeArray(timestamp(ta), rank_matrix, colnames(ta))
end

"""
    rowwise_competerank(ta::TimeArray) -> TimeArray

Compute "competition" ranks (also known as standard competition ranks or "1224" ranking) for each value in the `TimeArray` along rows, ignoring NaNs.

For each row, non-NaN values are ranked in ascending order (smallest gets rank 1, ties get the same rank, and the next rank is incremented by the number of tied values). NaN positions remain NaN in the output.

# Arguments
- `ta::TimeArray`: A `TimeArray` containing numerical data.

# Returns
- `TimeArray`: A new TimeArray with the same timestamps and column names as the input, but with values replaced by their row-wise competition ranks. NaN values are preserved in their original positions.

# Example
```julia
ta = TimeArray([1.0 3.0 NaN; 2.0 1.0 4.0], timestamp=1:2, colnames=[:A, :B, :C])
rowwise_competerank(ta)
```
"""
function rowwise_competerank(ta::TimeArray)
    values_matrix = values(ta)
    rank_matrix = similar(values_matrix, Float64)
    for i in axes(values_matrix, 1)
        row = values_matrix[i, :]
        non_nan_indices = findall(!isnan, row)
        non_nan_values = row[non_nan_indices]
        if !isempty(non_nan_values)
            row_ranks = competerank(non_nan_values)
            rank_matrix[i, non_nan_indices] .= row_ranks
        end
        # Preserve NaN values in rank matrix
        rank_matrix[i, setdiff(1:length(row), non_nan_indices)] .= NaN
    end
    return TimeArray(timestamp(ta), rank_matrix, colnames(ta))
end

"""
    rowwise_tiedrank(ta::TimeArray) -> TimeArray

Compute tied ranks for each value in the `TimeArray` along rows, ignoring NaNs.

For each row, non-NaN values are ranked in ascending order, with ties receiving the average of the ranks they span (tied ranking). NaN positions remain NaN in the output.

# Arguments
- `ta::TimeArray`: A `TimeArray` containing numerical data.

# Returns
- `TimeArray`: A new TimeArray with the same timestamps and column names as the input, but with values replaced by their row-wise tied ranks. NaN values are preserved in their original positions.

# Example
```julia
ta = TimeArray([1.0 3.0 NaN; 2.0 1.0 4.0], timestamp=1:2, colnames=[:A, :B, :C])
rowwise_tiedrank(ta)
```
"""
function rowwise_tiedrank(ta::TimeArray)
    values_matrix = values(ta)
    rank_matrix = similar(values_matrix, Float64)
    for i in axes(values_matrix, 1)
        row = values_matrix[i, :]
        non_nan_indices = findall(!isnan, row)
        non_nan_values = row[non_nan_indices]
        if !isempty(non_nan_values)
            row_ranks = StatsBase.tiedrank(non_nan_values)
            rank_matrix[i, non_nan_indices] .= row_ranks
        end
        # Preserve NaN values in rank matrix
        rank_matrix[i, setdiff(1:length(row), non_nan_indices)] .= NaN
    end
    return TimeArray(timestamp(ta), rank_matrix, colnames(ta))
end

"""
    rowwise_denserank(ta::TimeArray) -> TimeArray

Compute dense ranks for each value in the `TimeArray` along rows, ignoring NaNs.

For each row, non-NaN values are ranked in ascending order using dense ranking (no gaps in ranks for ties). NaN positions remain NaN in the output.

# Arguments
- `ta::TimeArray`: A `TimeArray` containing numerical data.

# Returns
- `TimeArray`: A new TimeArray with the same timestamps and column names as the input, but with values replaced by their row-wise dense ranks. NaN values are preserved in their original positions.

# Example
```julia
ta = TimeArray([1.0 3.0 NaN; 2.0 1.0 4.0], timestamp=1:2, colnames=[:A, :B, :C])
rowwise_denserank(ta)
```
"""
function rowwise_denserank(ta::TimeArray)
    values_matrix = values(ta)
    rank_matrix = similar(values_matrix, Float64)
    for i in axes(values_matrix, 1)
        row = values_matrix[i, :]
        non_nan_indices = findall(!isnan, row)
        non_nan_values = row[non_nan_indices]
        if !isempty(non_nan_values)
            row_ranks = StatsBase.denserank(non_nan_values)
            rank_matrix[i, non_nan_indices] .= row_ranks
        end
        # Preserve NaN values in rank matrix
        rank_matrix[i, setdiff(1:length(row), non_nan_indices)] .= NaN
    end
    return TimeArray(timestamp(ta), rank_matrix, colnames(ta))
end

################### PERCENTILE RANK FUNCTIONS ################################

"""
    rowwise_ordinal_pctrank(ta::TimeArray) -> TimeArray

Compute ordinal percentile ranks for each value in the `TimeArray` along rows, ignoring NaNs.

For each row, non-NaN values are ranked ordinally, and their percentile rank is calculated as (ordinal rank) / (maximum rank in the row). NaN values are preserved in their original positions.

# Arguments
- `ta::TimeArray`: TimeArray containing numerical data.

# Returns
- `TimeArray`: A new TimeArray with ordinal percentile ranks for each row.

# Notes
- Rows with only one non-NaN value will result in NaN for that row's percentile calculation.
- NaN values are preserved in their original positions.

# Example
```julia
ta = TimeArray([1.0 3.0 NaN; 2.0 1.0 4.0], timestamp=1:2, colnames=[:A, :B, :C])
rowwise_ordinal_pctrank(ta)
```
"""
function rowwise_ordinal_pctrank(ta::TimeArray)
    ta_ordrank = rowwise_ordinalrank(ta)
    max_values = [maximum([x for x in row if !isnan(x)]) for row in eachrow(values(ta_ordrank))]
    ta_ord_pct = ta_ordrank ./ max_values
    return ta_ord_pct
end


"""
    rowwise_tied_pctrank(ta::TimeArray) -> TimeArray

Compute tied percentile ranks for each value in the `TimeArray` along rows, ignoring NaNs.

For each row, non-NaN values are ranked using tied ranking, and their percentile rank is calculated as (tied rank) / (maximum rank in the row). NaN values are preserved in their original positions.

# Arguments
- `ta::TimeArray`: TimeArray containing numerical data.

# Returns
- `TimeArray`: A new TimeArray with tied percentile ranks for each row.

# Notes
- Rows with only one non-NaN value will result in NaN for that row's percentile calculation.
- NaN values are preserved in their original positions.

# Example
```julia
ta = TimeArray([1.0 3.0 NaN; 2.0 1.0 4.0], timestamp=1:2, colnames=[:A, :B, :C])
rowwise_tied_pctrank(ta)
```
"""
function rowwise_tied_pctrank(ta::TimeArray)
    ta_tiedrank = rowwise_tiedrank(ta)
    max_values = [maximum([x for x in row if !isnan(x)]) for row in eachrow(values(ta_tiedrank))]
    ta_tied_pct = ta_tiedrank ./ max_values
    return ta_tied_pct
end


################### QUANTILES ###################################

"""
    rowwise_ordinal_quantiles(ta::TimeArray, n_quantiles::Int=5) -> TimeArray

Assign each value in a TimeArray to a quantile (1 to n_quantiles) within its row, ignoring NaNs.

For each row, non-NaN values are assigned to quantiles based on their ordinal percentile rank. NaN values remain NaN.

# Arguments
- `ta::TimeArray`: Input TimeArray.
- `n_quantiles::Int`: Number of quantiles (default 5).

# Returns
- `TimeArray`: Same shape as input, with quantile numbers (1 = lowest, n_quantiles = highest).

# Example
```julia
ta = TimeArray([1.0 3.0 NaN; 2.0 1.0 4.0], timestamp=1:2, colnames=[:A, :B, :C])
rowwise_ordinal_quantiles(ta, 3)
```
"""
function rowwise_ordinal_quantiles(ta::TimeArray, n_quantiles::Int=5)
    ta_pctrank = rowwise_ordinal_pctrank(ta)
    values_matrix = values(ta_pctrank)
    result_matrix = fill(NaN, size(values_matrix))
    # Define quantile edges (0, 1/n, 2/n, ..., 1)
    quantile_edges = range(0, stop=1, length=n_quantiles+1)
    for row in axes(values_matrix, 1)
        row_data = values_matrix[row, :]
        for (col, value) in enumerate(row_data)
            if isnan(value)
                result_matrix[row, col] = NaN
            else
                # Find quantile bin (1-based)
                q = findfirst(i -> value <= quantile_edges[i], 2:length(quantile_edges))
                result_matrix[row, col] = isnothing(q) ? n_quantiles : q
            end
        end
    end
    return TimeArray(timestamp(ta), result_matrix, colnames(ta))
end

"""
    rowwise_tied_quantiles(ta::TimeArray, n_quantiles::Int=5) -> TimeArray

Assign each value in a TimeArray to a quantile (1 to n_quantiles) within its row, using tied percentile ranks, ignoring NaNs.

For each row, non-NaN values are assigned to quantiles based on their tied percentile rank. NaN values remain NaN.

# Arguments
- `ta::TimeArray`: Input TimeArray.
- `n_quantiles::Int`: Number of quantiles (default 5).

# Returns
- `TimeArray`: Same shape as input, with quantile numbers (1 = lowest, n_quantiles = highest).

# Example
```julia
ta = TimeArray([1.0 3.0 NaN; 2.0 1.0 4.0], timestamp=1:2, colnames=[:A, :B, :C])
rowwise_tiedquantiles(ta, 3)
```
"""
function rowwise_tied_quantiles(ta::TimeArray, n_quantiles::Int=5)
    ta_pctrank = rowwise_tied_pctrank(ta)
    values_matrix = values(ta_pctrank)
    result_matrix = fill(NaN, size(values_matrix))
    # Define quantile edges (0, 1/n, ..., 1)
    quantile_edges = range(0, stop=1, length=n_quantiles+1)
    for row in axes(values_matrix, 1)
        row_data = values_matrix[row, :]
        for (col, value) in enumerate(row_data)
            if isnan(value)
                result_matrix[row, col] = NaN
            else
                # Find quantile bin (1-based)
                q = findfirst(i -> value <= quantile_edges[i], 2:length(quantile_edges))
                result_matrix[row, col] = isnothing(q) ? n_quantiles : q
            end
        end
    end
    return TimeArray(timestamp(ta), result_matrix, colnames(ta))
end

######## COUNT FUNCTIONS #####################

"""
    rowwise_count(ta::TimeArray, target_value::Any) -> TimeArray

Count the occurrences of `target_value` in each row of a `TimeArray`.

# Arguments
- `ta::TimeArray`: The TimeArray to analyze.
- `target_value::Any`: The value to count within each row. If `target_value === NaN`, counts NaN values.

# Returns
- `TimeArray`: A new TimeArray where each entry represents the count of `target_value` (or NaN) in 
  the corresponding row of the input TimeArray. The timestamp from the input TimeArray is preserved.

# Example
```julia
ta = TimeArray([1 2 NaN; NaN 2 1; 1 1 1], timestamp=1:3, colnames=[:A, :B, :C])
rowwise_count(ta, 1)      # Counts 1s in each row
rowwise_count(ta, NaN)    # Counts NaNs in each row
```
"""
function rowwise_count(ta::TimeArray, target_value::Any)
    values_matrix = values(ta)
    if target_value === NaN
        counts = [count(isnan, row) for row in eachrow(values_matrix)]
        return TimeArray(timestamp(ta), counts, ["Count"])
    else
        counts = [count(x -> x == target_value, row) for row in eachrow(values_matrix)]
        return TimeArray(timestamp(ta), counts, ["Count"])
    end
end


"""
    rowwise_countall(ta::TimeArray) -> TimeArray

Count the occurrences of all unique (non-NaN) values in each row of a `TimeArray`.

# Arguments
- `ta::TimeArray`: The input TimeArray to analyze.

# Returns
- `TimeArray`: A new TimeArray where each column represents the count of a unique value (excluding NaN) in each row of the input. The timestamp is preserved.

# Example
```julia
ta = TimeArray([1 2 1; 2 2 1; 1 1 1], timestamp=1:3, colnames=[:A, :B, :C])
rowwise_countall(ta)
```
"""
function rowwise_countall(ta::TimeArray)
    # Get all unique, non-NaN values in the TimeArray
    values_range = ta |> values |> unique |> x -> filter(!isnan, x) |> sort

    # Prepare a DataFrame to collect results
    results_df = DataFrame(:TimeStamp => timestamp(ta))

    # For each unique value, count its occurrences rowwise
    for val in values_range
        results_df[!, "$val"] = rowwise_count(ta, val) |> values
    end

    # Return as a TimeArray, using the timestamp column
    return TimeArray(results_df, timestamp = :TimeStamp)
end


####### CONSECUTIVE COUNT #################
"""
    consecutive_values(ta::AbstractVector, target::Any, n::Int) -> Bool

Check if there are at least `n` consecutive occurrences of `target` in the vector `ta`.

# Arguments
- `ta::AbstractVector`: The input vector to search.
- `target::Any`: The value to look for consecutive occurrences of.
- `n::Int`: The number of consecutive occurrences required.

# Returns
- `Bool`: Returns `true` if `target` appears at least `n` times in a row in `ta`, otherwise `false`.

# Example
```julia
ta = [0, 1, 1, 1, 0, 1, 1]
consecutive_values(ta, 1, 3) # returns true
consecutive_values(ta, 0, 2) # returns false
```
"""
function consecutive_values(ta::AbstractVector, target::Any, n::Int)
    count = 0
    for v in ta
        if v == target
            count += 1
            if count == n
                return true
            end
        else
            count = 0
        end
    end
    return false
end

"""
    consecutive_values(mtx::AbstractMatrix, target::Any, n::Int) -> Vector{Bool}

Check each column of the matrix `mtx` for at least `n` consecutive occurrences of `target`.

# Arguments
- `mtx::AbstractMatrix`: The input matrix to search (columns are checked independently).
- `target::Any`: The value to look for consecutive occurrences of.
- `n::Int`: The number of consecutive occurrences required.

# Returns
- `Vector{Bool}`: A boolean vector where each element indicates if the corresponding column contains at least `n` consecutive `target` values.

# Example
```julia
mtx = [0 1; 1 1; 1 0; 1 1]
consecutive_values(mtx, 1, 3) # returns [true, true]
```
"""
function consecutive_values(mtx::AbstractMatrix, target::Any, n::Int)
    results = Vector{Bool}(undef, size(mtx, 2))
    for (i, col) in enumerate(eachcol(mtx))
        results[i] = consecutive_values(col, target, n)
    end
    return results
end

"""
    consecutive_values(ta::TimeArray, target::Any, n::Int) -> Vector{Bool}

Check each column of the `TimeArray` for at least `n` consecutive occurrences of `target`.

# Arguments
- `ta::TimeArray`: The input TimeArray to search (columns are checked independently).
- `target::Any`: The value to look for consecutive occurrences of.
- `n::Int`: The number of consecutive occurrences required.

# Returns
- `Vector{Bool}`: A boolean vector where each element indicates if the corresponding column contains at least `n` consecutive `target` values.

# Example
```julia
consecutive_values(returns_ts, 0, 10)
```
"""
function consecutive_values(ta::TimeArray, target::Any, n::Int)
    return consecutive_values(values(ta), target, n)
end



################################### THE END ######################################################################################