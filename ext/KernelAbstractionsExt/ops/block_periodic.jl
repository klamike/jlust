# ─── Block-Toeplitz periodic + BSM-with-patches SpMV ────────────────────────
#
# A block-banded matrix with one diagonal block CSR shared across T periods +
# one off-diagonal CSR pair shared across the T-1 transitions can be SpMV'd
# without ever materialising the T-fold replica.  Each thread owns one output
# row tid; from tid we recover (period_t, in-period row, kind) and walk the
# shared CSRs with a per-period column offset.
#
# Block-level "selector patches" — (Dense, ShiftedDiag)-shaped blocks
# embedded in the BSM — are walked alongside the CSR in the same fused
# kernel.  Each patch contributes a single per-row `acc += val * x[col]` term
# without any indirect load (val and the col offset are kernel literals from
# the patch tuple).  Tuple length is type-known so the loop unrolls.
#
# Both kernels are `@kernel` here so every backend KA targets (CUDA, ROCm,
# CPU, POCL, oneAPI) runs the same structurally-aware SpMV.  CUDA-specific
# tricks (LDG read-only data cache via `Base.Experimental.Const`) are gated
# by the `_supports_ldg(ka)` trait so each backend opts in to its own
# hardware features.

# Helper: apply the tuple of selector patches to row r, returning the
# accumulated patch contribution.  Tuple unrolls (length is type-known).
@inline function _apply_patches(patches::Tuple, r::Int32,
                                  col_offset_extra::Int32, x, ::Type{T}) where T
    acc = zero(T)
    Base.@nexprs 0 _ -> nothing  # placeholder so @nexprs isn't reached without an N
    # Rely on Julia's tuple-unroll in the for-loop.  Each iteration is fully
    # specialised on the patch's static type, so `patch.val::T` is a literal.
    for patch in patches
        if r >= patch.row_start && r <= patch.row_end
            col  = (r - patch.row_start + Int32(1)) + patch.col_offset + col_offset_extra
            acc += patch.val * x[col]
        end
    end
    acc
end

# ── BSM SpMV: assembled CSR + selector patches ────────────────────────────────
#
# The classic BlockSparseMatrix SpMV routes through the assembled CSR.  When
# some blocks are constant-scaled identities (ShiftedDiag-formatted), they're
# extracted as patches at compile time — the kernel walks the leaner CSR plus
# the patches in one fused pass.

@kernel inbounds=true function _bsm_with_patches_spmv_kernel!(
        asm_pos, asm_crd, asm_nzval,
        _y, x_raw,
        n_total_rows::Int32, patches::Tuple,
        ::Val{ZERO_BETA}, beta,
        ::Val{LDG}) where {ZERO_BETA, LDG}
    T   = eltype(_y)
    tid = @index(Global, Linear)
    if tid <= n_total_rows
        x = LDG ? Base.Experimental.Const(x_raw) : x_raw
        r = Int32(tid)
        acc = zero(T)
        lo = asm_pos[r]
        hi = asm_pos[r + Int32(1)] - Int32(1)
        for k in lo:hi
            acc += asm_nzval[k] * x[asm_crd[k]]
        end
        acc += _apply_patches(patches, r, Int32(0), x, T)
        _y[tid] = ZERO_BETA ? acc : acc + beta * _y[tid]
    end
end

# Row layout (1-based) for the periodic BBM kernel below: period t occupies
#   [(t-1)*P+1 .. (t-1)*P+n_diag]   (diag rows of period t),  P = n_diag + n_off
#   [(t-1)*P+n_diag+1 .. t*P]       (off rows for t→t+1, only if t < T_per).
# Total rows: T_per*n_diag + (T_per-1)*n_off.

@kernel inbounds=true function _bbm_periodic_spmv_kernel!(
        d_pos, d_crd, d_nzval,
        n_pos, n_crd, n_nzval,
        p_pos, p_crd, p_nzval,
        _y, x_raw,
        n_diag::Int32, n_off::Int32, n_cols::Int32,
        n_total_rows::Int32,
        patches::Tuple,
        ::Val{ZERO_BETA}, beta,
        ::Val{LDG}) where {ZERO_BETA, LDG}
    T   = eltype(_y)
    tid = @index(Global, Linear)
    if tid <= n_total_rows
        x = LDG ? Base.Experimental.Const(x_raw) : x_raw

        period_size  = n_diag + n_off
        period_m1    = (Int32(tid) - Int32(1)) ÷ period_size
        rem          = (Int32(tid) - Int32(1)) - period_m1 * period_size
        col_off_t    = period_m1 * n_cols

        acc = zero(T)

        if rem < n_diag
            r = rem + Int32(1)
            lo = d_pos[r]
            hi = d_pos[r + Int32(1)] - Int32(1)
            for k in lo:hi
                col  = d_crd[k] + col_off_t
                acc += d_nzval[k] * x[col]
            end
            acc += _apply_patches(patches, r, col_off_t, x, T)
        else
            r = rem - n_diag + Int32(1)
            col_off_next = (period_m1 + Int32(1)) * n_cols
            lo = n_pos[r]
            hi = n_pos[r + Int32(1)] - Int32(1)
            for k in lo:hi
                col  = n_crd[k] + col_off_t
                acc += n_nzval[k] * x[col]
            end
            lo = p_pos[r]
            hi = p_pos[r + Int32(1)] - Int32(1)
            for k in lo:hi
                col  = p_crd[k] + col_off_next
                acc += p_nzval[k] * x[col]
            end
        end

        _y[tid] = ZERO_BETA ? acc : acc + beta * _y[tid]
    end
end

# Specialised path for "constant-value selector off-diag": when the (neg, pos)
# pair encodes a scaled identity (1 nnz/row at col=row), the kernel drops
# pos/crd/nzval reads on off-diag rows entirely — y[off_row] = neg·x[r+col_t]
# + pos·x[r+col_next].  Saves 8 indirect loads per off-diag row vs the
# generic kernel above.

@kernel inbounds=true function _bbm_periodic_selector_kernel!(
        d_pos, d_crd, d_nzval,
        _y, x_raw,
        n_diag::Int32, n_off::Int32, n_cols::Int32,
        n_total_rows::Int32,
        neg_val, pos_val,
        patches::Tuple,
        ::Val{ZERO_BETA}, beta,
        ::Val{LDG}) where {ZERO_BETA, LDG}
    T   = eltype(_y)
    tid = @index(Global, Linear)
    if tid <= n_total_rows
        x = LDG ? Base.Experimental.Const(x_raw) : x_raw

        period_size  = n_diag + n_off
        period_m1    = (Int32(tid) - Int32(1)) ÷ period_size
        rem          = (Int32(tid) - Int32(1)) - period_m1 * period_size
        col_off_t    = period_m1 * n_cols

        if rem < n_diag
            r = rem + Int32(1)
            acc = zero(T)
            lo = d_pos[r]
            hi = d_pos[r + Int32(1)] - Int32(1)
            for k in lo:hi
                col  = d_crd[k] + col_off_t
                acc += d_nzval[k] * x[col]
            end
            acc += _apply_patches(patches, r, col_off_t, x, T)
            _y[tid] = ZERO_BETA ? acc : acc + beta * _y[tid]
        else
            r = rem - n_diag + Int32(1)
            col_off_next = (period_m1 + Int32(1)) * n_cols
            acc = T(neg_val) * x[r + col_off_t] + T(pos_val) * x[r + col_off_next]
            _y[tid] = ZERO_BETA ? acc : acc + beta * _y[tid]
        end
    end
end

# Public launch entry points — declared in JLUST core, implemented here.
# Backend-trait gating (`_supports_ldg`, `_default_workgroup_size`) happens at
# the launch site so each backend gets the right workgroup size and LDG
# opt-in for free.

function JLUST._bsm_with_patches_spmv_launch!(ka,
        asm_pos, asm_crd, asm_nzval,
        y, x,
        n_total_rows::Int32, patches::Tuple,
        zero_beta::Bool, beta)
    ws = JLUST._default_workgroup_size(ka)
    _bsm_with_patches_spmv_kernel!(ka, ws)(
        asm_pos, asm_crd, asm_nzval,
        y, x,
        n_total_rows, patches,
        Val(zero_beta), beta,
        Val(JLUST._supports_ldg(ka));
        ndrange = Int(n_total_rows))
    return y
end

function JLUST._bbm_periodic_spmv_launch!(ka,
        d_pos, d_crd, d_nzval,
        n_pos, n_crd, n_nzval,
        p_pos, p_crd, p_nzval,
        y, x,
        n_diag::Int32, n_off::Int32, n_cols::Int32, n_total_rows::Int32,
        patches::Tuple,
        zero_beta::Bool, beta)
    ws = JLUST._default_workgroup_size(ka)
    _bbm_periodic_spmv_kernel!(ka, ws)(
        d_pos, d_crd, d_nzval,
        n_pos, n_crd, n_nzval,
        p_pos, p_crd, p_nzval,
        y, x,
        n_diag, n_off, n_cols, n_total_rows,
        patches,
        Val(zero_beta), beta,
        Val(JLUST._supports_ldg(ka));
        ndrange = Int(n_total_rows))
    return y
end

function JLUST._bbm_periodic_selector_launch!(ka,
        d_pos, d_crd, d_nzval,
        y, x,
        n_diag::Int32, n_off::Int32, n_cols::Int32, n_total_rows::Int32,
        neg_val, pos_val,
        patches::Tuple,
        zero_beta::Bool, beta)
    ws = JLUST._default_workgroup_size(ka)
    _bbm_periodic_selector_kernel!(ka, ws)(
        d_pos, d_crd, d_nzval,
        y, x,
        n_diag, n_off, n_cols, n_total_rows,
        neg_val, pos_val,
        patches,
        Val(zero_beta), beta,
        Val(JLUST._supports_ldg(ka));
        ndrange = Int(n_total_rows))
    return y
end
