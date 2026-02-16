module Luminal

using SymbolicUtils
using SymbolicUtils: Sym, BasicSymbolic

# Type alias for dimension values: either a concrete Int or a symbolic expression
const DimType = Union{Int, BasicSymbolic{Int}}

# Shape Tracking
include("ShapeTracker.jl")

# Core Data Structures
include("Ops.jl")
include("Graph.jl")

# Graph Construction API
include("HighLevelOps.jl")

# Execution Engine
include("Execution.jl")

# SymbolicUtils.jl Integration (graph <-> symbolic expression conversion)
include("MetatheoryIntegration.jl")

# Compiler (optimization rules using SymbolicUtils.jl)
include("Compiler.jl")

end # module Luminal
