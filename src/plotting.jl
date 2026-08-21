####### PLOTS FOR ANALYSIS ##########################################################

"""
    plot_factor_distribution(factors::TimeArray; bins::Int = 20) -> Plot

Plot the mean factor score over time alongside its distribution.

The two views answer different questions: the time series shows whether the factor drifts
or spikes, while the histogram shows whether its scores are well spread or bunched up.

# Arguments
- `factors::TimeArray`: Factor scores, one column per security.
- `bins::Int`: Number of bins for the histogram (default: 20).

# Returns
- A 2x1 plot: the cross-sectional mean score per timestamp above, its histogram below.

# Throws
- `ArgumentError`: If `factors` is empty.
"""
function plot_factor_distribution(factors::TimeArray; bins::Int = 20)

    # Ensure the factor_score is not empty
    if isempty(factors)
        throw(ArgumentError("factor_score cannot be empty"))
    end

    # Calculate the row mean
    mu_row = row_mean(factors)

    # Time series plot
    p1 = Plots.plot(timestamp(mu_row), values(mu_row), 
    title = "Mean Factor Values Over Time",
    label = "",
    xlabel = "Time",
    ylabel = "Mean Factor Score",
    titlefontsize = 12,
    legend = :top)

    # Histogram plot
    mu_row_values = values(mu_row)
    p2 = Plots.histogram(mu_row_values, 
    title = "Distribution of Mean Factor Scores",
    label = "",
    xlabel = "Mean Factor Score",
    ylabel = "Frequency",
    titlefontsize = 12,
    legend = :top,
    bins = bins)

    # Combine plots into a 2x1 subplot
    return Plots.plot(p1, p2, layout = (2, 1), size = (800,600))
end


"""
    plot_performance_table(table::DataFrame) -> Plot

Plot a bar chart per headline statistic of a summary performance table.

# Arguments
- `table::DataFrame`: A table as returned by `quantile_performance_table` or
    `performance_table`: a leading `Stat` column naming each row, then one column
    per quantile or asset.

# Returns
- A 3x2 grid of bar charts, one per plotted statistic.
"""
function plot_performance_table(table::DataFrame)

    quantile_names = names(table)[2:end]
    bar_width = 0.3

    # Rows are found by name rather than by position, so the plot stays correct
    # whatever order the table lists its statistics in.
    function bar_for(stat)
        row = findfirst(==(stat), table.Stat)
        isnothing(row) && throw(ArgumentError("Performance table has no \"$stat\" row"))
        return Plots.bar(quantile_names, collect(table[row, 2:end]), labels = stat, bar_width = bar_width)
    end

    stats_to_plot = ["Annual Return", "Annual StDev", "Sharpe Ratio",
    "Max Drawdowns", "Up Capture", "Down Capture"]

    return Plots.plot(bar_for.(stats_to_plot)..., layout = (3, 2), size = (800,600), legend=:outertop)
end
