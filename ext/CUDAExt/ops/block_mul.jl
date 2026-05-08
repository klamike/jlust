# ─── BlockSparseMatrix GPU mul! with lazy CUDA graph capture ─────────────────
#
# First call with a given (A, y, x) triple: JIT warm-up on the default stream,
# then capture into a CUDA graph, instantiate, and cache by
# (pointer(y), pointer(x)).  Subsequent calls: CUDA.launch(exec) — no kernel-
# launch overhead on the host, graph replays on the GPU.
#
# During external graph capture (CUDA.is_capturing()): sequential path only
# (CUDA does not allow nested captures).
#
# Cache invalidation: update_block_values! with a type-mismatching new_nzval
# creates a new USTensor at a new CuArray pointer.  The cached graph would then
# reference a stale (possibly freed) allocation.  The CUDAExt override below
# clears A's cache entry before delegating to the core method.
#
# In-place update_block_values! (same concrete type, copyto! path) keeps the
# same pointer, so the cached graph automatically picks up the new values —
# the graph captured the pointer, not the data.

import LinearAlgebra
import JLUST: BlockSparseMatrix

# Cache: A → Dict{(ptr_y, ptr_x), CuGraphExec}
const _bm_graph_cache = IdDict{BlockSparseMatrix, Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}}()

function _bm_mul_seq!(y::CuVector, A::BlockSparseMatrix, x::CuVector, be, nb_r::Int, nb_c::Int)
    for i in 1:nb_r
        y_sl      = view(y, A._row_off[i]+1 : A._row_off[i+1])
        first_col = true
        for j in 1:nb_c
            b = A.blocks[i, j]
            b === nothing && continue
            x_sl = view(x, A._col_off[j]+1 : A._col_off[j+1])
            β    = first_col ? false : true
            JLUST.execute(JLUST.SpMVOp, b, JLUST.ust(x_sl), JLUST.ust(y_sl); backend=be, beta=β)
            first_col = false
        end
    end
    return y
end

function LinearAlgebra.mul!(y::CuVector, A::BlockSparseMatrix, x::CuVector;
                              backend::Union{AbstractUSTBackend,Nothing}=nothing)
    nb_r, nb_c = size(A.blocks)
    be = something(backend, EmitterBackend())

    # During external capture: sequential only (nested capture not allowed).
    CUDA.is_capturing() && return _bm_mul_seq!(y, A, x, be, nb_r, nb_c)

    key   = (UInt(pointer(y)), UInt(pointer(x)))
    cache = get!(() -> Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}(), _bm_graph_cache, A)

    if haskey(cache, key)
        CUDA.launch(cache[key])
        return y
    end

    # JIT warm-up: run once on the default stream to compile all kernels.
    _bm_mul_seq!(y, A, x, be, nb_r, nb_c)
    CUDA.synchronize()

    # Capture the same sequential path into a graph and cache the executable.
    g = CUDA.capture() do
        _bm_mul_seq!(y, A, x, be, nb_r, nb_c)
    end
    cache[key] = CUDA.instantiate(g)
    # y already holds the result from the warm-up run; return without re-launching.
    return y
end

# Override update_block_values! for CuVector to clear the graph cache whenever
# the value pointer changes (type-mismatch path creates a new CuArray).
function JLUST.update_block_values!(A::BlockSparseMatrix, i::Int, j::Int, new_nzval::CuVector)
    b = A.blocks[i, j]
    if b isa USTensor && typeof(new_nzval) !== typeof(nonzeros(b))
        delete!(_bm_graph_cache, A)
    end
    invoke(JLUST.update_block_values!, Tuple{BlockSparseMatrix, Int, Int, AbstractVector},
           A, i, j, new_nzval)
end
