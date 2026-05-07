# ─── EmitterBackend: sparse triangular solve (multiple RHS) ──────────────────
#
# Same sequential-dependency constraint as SpSV — see spsv.jl.
# CUSPARSEBackend implements SpSM via cusparseSpSM.

JLUST.supports_backend(::EmitterBackend, ::SpSMOp) = false

function JLUST.sparse_sm!(::EmitterBackend, u_A::USTensor, u_B::USTensor, u_C::USTensor; kw...)
    error("EmitterBackend does not support sparse_sm! — use CUSPARSEBackend() instead.")
end
