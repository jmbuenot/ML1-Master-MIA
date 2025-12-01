import Pkg
using Pkg
Pkg.activate(".") 
using CSV
using DataFrames

# Load dataset
df = CSV.read("datasets/OnlineNewsPopularity.csv", DataFrame)

println("Total columns in dataset: $(ncol(df))")
println("\nAll column names:")
for (i, col) in enumerate(names(df))
    println("  $i. $col")
end

# Apply same logic as main.jl
all_cols = names(df)

# Find the exact target column
target_col_idx = findfirst(col -> strip(col) == "shares", all_cols)
target_col_name = target_col_idx !== nothing ? all_cols[target_col_idx] : nothing

println("\n\nTarget column index: $target_col_idx")
println("Target column: '$target_col_name'")

# Filter features
feature_cols = filter(col -> 
    strip(col) != "url" && 
    strip(col) != "timedelta" && 
    col != target_col_name,
    all_cols)

println("\nNumber of feature columns: $(length(feature_cols))")
println("\nFeature columns:")
for (i, col) in enumerate(feature_cols)
    println("  $i. $col")
end

# Check for self-reference columns
println("\n\nSelf-reference columns (should be included):")
for col in feature_cols
    if contains(col, "self_reference")
        println("  ✓ $col")
    end
end

