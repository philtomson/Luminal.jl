module Luminal

# Symbolic Expression Engine
include("Symbolic.jl")
using .Symbolic

# Shape Tracking
include("ShapeTracker.jl")

# Core Data Structures
include("Ops.jl")
include("Graph.jl")

# Graph Construction API
include("HighLevelOps.jl")

# Compiler
# Symbolics.jl Integration
include("SymbolicsIntegration.jl")

include("Compiler.jl")

# Execution Engine
include("Execution.jl")

export SymbolicsIntegration # Export the new submodule

end # module Luminal
