# ─── EmitterBackend: sparse triangular solve (single RHS) ────────────────────
#
# Triangular solve requires sequential forward/backward substitution — each row
# depends on all previously solved rows. This is not expressible as a
# work-parallel scatter/gather kernel without level-set (wavefront) analysis,
# which is out of scope for the current emitter.
#
# CUSPARSEBackend implements SpSV via cusparseSpSV.

JLUST.supports_backend(::EmitterBackend, ::SpSVOp) = false

function JLUST.sparse_sv!(::EmitterBackend, u_A::USTensor, u_b::USTensor, u_x::USTensor; kw...)
    error("EmitterBackend does not support sparse_sv! — use CUSPARSEBackend() instead.")
end
