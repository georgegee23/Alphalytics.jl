

####### TIME ARRAY MANIPULATIONS ##########################################################

function row_mean(ta::TimeArray)

    """
    Calculate the mean of each row in a TimeArray, ignoring NaN values.

    Parameters:
    - ta: TimeArray containing data with possible NaN values

    Returns:
    - TimeArray with the same timestamps as the input, where each value represents 
      the mean of non-NaN values in the corresponding row of the input.
    """
    
    ta_mtx = values(ta)
    
    # Calculate row means, ignoring NaN values
    row_mu = [mean(filter(!isnan, row)) for row in eachrow(ta_mtx)]
    
    # Create a meaningful column name
    col_name = [:Mean]
    
    return TimeArray(timestamp(ta), row_mu, col_name)
end


function column_mean(ta::TimeArray)

    """
    Calculate the mean of each column in a TimeArray, ignoring NaN values.

    Parameters:
    - ta: TimeArray containing data with possible NaN values

    Returns:
    - DataFrame with the column name, where each value represents the mean of non-NaN values.
    """
    
    ta_mtx = values(ta)
    
    # Calculate row means, ignoring NaN values
    col_mu = [mean(filter(!isnan, col)) for col in eachcol(ta_mtx)]
    
    return DataFrame(ID = colnames(ta), Mean = col_mu)
end


function rowwise_ordinalrank(ta::TimeArray)

    """
    Compute ordinal ranks for each value in the `TimeArray` along rows, ignoring `NaN`s.

        This function processes each row of the input `TimeArray` independently. For each row, it assigns ranks to the non-`NaN` values based on their order: the smallest value gets rank 1, the next smallest gets rank 2, and so on. Positions with `NaN` in the input remain `NaN` in the output.
        
        # Parameters
        - `ta::TimeArray`: A `TimeArray` containing numerical data. It is expected to have a 2D matrix of values, with rows typically representing time points and columns representing variables.
        
        # Returns
        - A new `TimeArray` with the same timestamps and column names as the input, but with values replaced by their row-wise ordinal ranks. `NaN` values are preserved in their original positions.
        
        # Notes
        - It also assumes that the `TimeArray` is from the `TimeSeries.jl` package or a similar library that provides `values`, `timestamp`, and `colnames` functions.
    """
    
    # Extract values from TimeArray
    values_matrix = values(ta)
    
    # Initialize matrix for ranks
    rank_matrix = similar(values_matrix, Float64)
    
    # Compute rowwise ordinal ranks
    for i in axes(values_matrix,1)
        row = values_matrix[i, :]
        non_nan_indices = findall(!isnan, row)
        non_nan_values = row[non_nan_indices]
        
        if !isempty(non_nan_values)
            # Compute ordinal ranks for non-NaN values
            row_ranks = ordinalrank(non_nan_values)
            
            # Assign ranks back to the original indices
            rank_matrix[i, non_nan_indices] .= row_ranks
        end
        
        # Preserve NaN values in rank matrix
        rank_matrix[i, setdiff(1:end, non_nan_indices)] .= NaN
    end
    
    # Create new TimeArray with computed ranks
    return TimeArray(timestamp(ta), rank_matrix, colnames(ta))
end

function rowwise_competerank(ta::TimeArray)
    """
    Compute ordinal ranks for each value in the TimeArray along rows, ignoring NaNs.

    Parameters:
    - ta: TimeArray containing numerical data.

    Returns:
    - A TimeArray with ordinal ranks computed for each row.
    """
    
    # Extract values from TimeArray
    values_matrix = values(ta)
    
    # Initialize matrix for ranks
    rank_matrix = similar(values_matrix, Float64)
    
    # Compute rowwise ordinal ranks
    for i in axes(values_matrix, 1)
        row = values_matrix[i, :]
        non_nan_indices = findall(!isnan, row)
        non_nan_values = row[non_nan_indices]
        
        if !isempty(non_nan_values)
            # Compute ordinal ranks for non-NaN values
            row_ranks = competerank(non_nan_values)
            
            # Assign ranks back to the original indices
            rank_matrix[i, non_nan_indices] .= row_ranks
        end
        
        # Preserve NaN values in rank matrix
        rank_matrix[i, setdiff(1:end, non_nan_indices)] .= NaN
    end
    
    # Create new TimeArray with computed ranks
    return TimeArray(timestamp(ta), rank_matrix, colnames(ta))
end

function rowwise_tiedrank(ta::TimeArray)
    """
    Compute tied ranks for each value in the TimeArray along rows, ignoring NaNs.

    Parameters:
    - ta: TimeArray containing numerical data.

    Returns:
    - A TimeArray with ordinal ranks computed for each row.
    """
    
    # Extract values from TimeArray
    values_matrix = values(ta)
    
    # Initialize matrix for ranks
    rank_matrix = similar(values_matrix, Float64)
    
    # Compute rowwise ordinal ranks
    for i in axes(values_matrix, 1)
        row = values_matrix[i, :]
        non_nan_indices = findall(!isnan, row)
        non_nan_values = row[non_nan_indices]
        
        if !isempty(non_nan_values)
            # Compute ordinal ranks for non-NaN values
            row_ranks = StatsBase.tiedrank(non_nan_values)
            
            # Assign ranks back to the original indices
            rank_matrix[i, non_nan_indices] .= row_ranks
        end
        
        # Preserve NaN values in rank matrix
        rank_matrix[i, setdiff(1:end, non_nan_indices)] .= NaN
    end
    
    # Create new TimeArray with computed ranks
    return TimeArray(timestamp(ta), rank_matrix, colnames(ta))
end

function rowwise_denserank(ta::TimeArray)
    """
    Compute tied ranks for each value in the TimeArray along rows, ignoring NaNs.

    Parameters:
    - ta: TimeArray containing numerical data.

    Returns:
    - A TimeArray with ordinal ranks computed for each row.
    """
    
    # Extract values from TimeArray
    values_matrix = values(ta)
    
    # Initialize matrix for ranks
    rank_matrix = similar(values_matrix, Float64)
    
    # Compute rowwise ordinal ranks
    for i in axes(values_matrix, 1)
        row = values_matrix[i, :]
        non_nan_indices = findall(!isnan, row)
        non_nan_values = row[non_nan_indices]
        
        if !isempty(non_nan_values)
            # Compute ordinal ranks for non-NaN values
            row_ranks = StatsBase.denserank(non_nan_values)
            
            # Assign ranks back to the original indices
            rank_matrix[i, non_nan_indices] .= row_ranks
        end
        
        # Preserve NaN values in rank matrix
        rank_matrix[i, setdiff(1:end, non_nan_indices)] .= NaN
    end
    
    # Create new TimeArray with computed ranks
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

function rowwise_quantiles(ta::TimeArray, n_quantiles::Int=5)
    
    """
    Convert values in a TimeArray to quantiles rowwise, ignoring NaN values.

    Parameters:
    - ta: Input TimeArray
    - n_quantiles: Number of quantiles (default: 5)

    Returns:
    - A new TimeArray with non-NaN values replaced by their quantile ranks
    """
    
    ta_pctrank = rowwise_ordinal_pctrank(ta)

    values_matrix = values(ta_pctrank)
    result_matrix = similar(values_matrix)
    
    for row in axes(values_matrix, 1)
        row_data = values_matrix[row, :]
        non_nan_data = filter(!isnan, row_data)
        
        if isempty(non_nan_data)
            result_matrix[row, :] .= NaN
        else
            quantile_breaks = quantile(non_nan_data, (0:n_quantiles)/n_quantiles)
            
            for (col, value) in enumerate(row_data)
                if isnan(value)
                    result_matrix[row, col] = NaN
                else
                    result_matrix[row, col] = findfirst(q -> value <= q, quantile_breaks[2:end])
                end
            end
        end
    end
    
    return TimeArray(timestamp(ta), result_matrix, colnames(ta))
end

function rowwise_tiedquantiles(ta::TimeArray, n_quantiles::Int=5)
    
    """
    Convert values in a TimeArray to quantiles rowwise, ignoring NaN values.

    Parameters:
    - ta: Input TimeArray
    - n_quantiles: Number of quantiles (default: 5)

    Returns:
    - A new TimeArray with non-NaN values replaced by their quantile ranks
    """
    
    ta_pctrank = rowwise_tied_pctrank(ta)

    values_matrix = values(ta_pctrank)
    result_matrix = similar(values_matrix)
    
    for row in axes(values_matrix, 1)
        row_data = values_matrix[row, :]
        non_nan_data = filter(!isnan, row_data)
        
        if isempty(non_nan_data)
            result_matrix[row, :] .= NaN
        else
            quantile_breaks = quantile(non_nan_data, (0:n_quantiles)/n_quantiles)
            
            for (col, value) in enumerate(row_data)
                if isnan(value)
                    result_matrix[row, col] = NaN
                else
                    result_matrix[row, col] = findfirst(q -> value <= q, quantile_breaks[2:end])
                end
            end
        end
    end
    
    return TimeArray(timestamp(ta), result_matrix, colnames(ta))
end


function rowwise_count(ta::TimeArray, target_value::Any)

    """
    Counts the occurrences of `target_value` in each row of the given `TimeArray` `ta`.

    # Arguments
    - `ta::TimeArray`: The TimeArray to analyze.
    - `target_value::Any`: The value to count within each row.

    # Returns
    - A new `TimeArray` where each entry represents the count of `target_value` in 
    the corresponding row of the input `TimeArray`. The timestamp from the input 
    `TimeArray` is preserved.
    """

    values_matrix = values(ta)
    counts = [count(x -> x == target_value, row) for row in eachrow(values_matrix)]
    
    return TimeArray(timestamp(ta), counts, ["CountOf_$target_value"])
end




function rowwise_countall(ta::TimeArray)



    values_range = ta |> values |> unique |> x -> filter(!isnan, x) |> sort
    results_df = DataFrame(:TimeStamp => timestamp(ta))

    for val in values_range

        results_df[!, "Var_$val"] = rowwise_count(ta, val) |> values 
    end

    return TimeArray(timestamp = :TimeStamp, results_df)
end

############################# DATA CLEANING ########################################

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