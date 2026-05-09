# ─── Abstract type ────────────────────────────────────────────────────────────

abstract type AbstractLevelFormat end

# ─── Concrete level formats ───────────────────────────────────────────────────

# Parametric on dimension extent: `Sz::Int` when the extent is known statically
# (e.g. blocked inner dims in BSR), `nothing` when the extent is the matrix-shape-
# dependent runtime value.  Static extents enable compile-time loop bounds in the
# emitter and let formats that differ only in block size (BSR(2,2) vs BSR(4,4))
# carry that distinction in the type system.
struct DenseLevel{Sz} <: AbstractLevelFormat end
struct BatchLevel{Sz} <: AbstractLevelFormat end
DenseLevel()           = DenseLevel{nothing}()
DenseLevel(sz::Int)    = (sz > 0 ? DenseLevel{sz}() :
    throw(InvalidTensorFormat("DenseLevel size must be positive, got $sz")))
BatchLevel()           = BatchLevel{nothing}()
BatchLevel(sz::Int)    = (sz > 0 ? BatchLevel{sz}() :
    throw(InvalidTensorFormat("BatchLevel size must be positive, got $sz")))
dense_size(::DenseLevel{Sz}) where {Sz} = Sz
batch_size(::BatchLevel{Sz}) where {Sz} = Sz

struct SingletonLevel <: AbstractLevelFormat end
struct RangeLevel     <: AbstractLevelFormat end

# Parametric on Unique/Ordered Bools so format-level structure (e.g. CSR vs COO) is
# captured in the type system — kernel dispatch finds the right method without a
# value-level branch on level fields.
struct CompressedLevel{Unique,Ordered} <: AbstractLevelFormat end
CompressedLevel(; unique::Bool=true, ordered::Bool=true) = CompressedLevel{unique,ordered}()

# Parametric on bit width.  All instances of DeltaLevel{B} are singletons.
struct DeltaLevel{Bits} <: AbstractLevelFormat
    DeltaLevel{Bits}() where {Bits} = (Bits isa Int && Bits > 0) ? new{Bits}() :
        throw(InvalidTensorFormat("DeltaLevel bits must be a positive Int, got $Bits"))
end
DeltaLevel(bits::Int) = DeltaLevel{bits}()

# Bit-width accessor for runtime use (Composer / display).
delta_bits(::DeltaLevel{B}) where {B} = B

# Implicit shifted scaled-identity level: row r has exactly one nonzero at
# column `r + Shift`, all with the constant value `Val`.  No nzval / pos / crd
# arrays — both Shift and Val live in the type system, so kernels emitted
# against this level read the value as a literal (zero memory traffic for the
# block's structure).  Used as the "selector block" abstraction in
# BlockSparseMatrix: a (Dense, ShiftedDiag) block is recognized at compile
# time and inlined as a per-row `acc += Val * x[col]` term — no CSR replication.
#
# Both type parameters are bits-types so the level remains a singleton (zero
# runtime fields), letting the @generated walker reconstruct it via `L()`.
struct ShiftedDiagLevel{Shift, Val} <: AbstractLevelFormat
    function ShiftedDiagLevel{Shift, Val}() where {Shift, Val}
        Shift isa Integer || throw(InvalidTensorFormat(
            "ShiftedDiagLevel Shift must be Integer, got $Shift::$(typeof(Shift))"))
        isbits(Val) || throw(InvalidTensorFormat(
            "ShiftedDiagLevel Val must be a bits-type literal, got $Val::$(typeof(Val))"))
        new{Shift, Val}()
    end
end
ShiftedDiagLevel(; shift::Integer=0, val=1.0) = ShiftedDiagLevel{Int(shift), val}()
diag_shift(::ShiftedDiagLevel{S, V}) where {S, V} = S
diag_val(::ShiftedDiagLevel{S, V}) where {S, V}   = V

Base.hash(::DenseLevel{Sz}, h::UInt) where {Sz} = hash(Sz, hash(:DenseLevel, h))
Base.hash(::BatchLevel{Sz}, h::UInt) where {Sz} = hash(Sz, hash(:BatchLevel, h))
Base.hash(::SingletonLevel, h::UInt)            = hash(:SingletonLevel, h)
Base.hash(::RangeLevel,     h::UInt)            = hash(:RangeLevel,     h)
Base.hash(::CompressedLevel{U,O}, h::UInt) where {U,O} =
    hash(U, hash(O, hash(:CompressedLevel, h)))
Base.hash(::DeltaLevel{B},        h::UInt) where {B}   =
    hash(B, hash(:DeltaLevel, h))
Base.hash(::ShiftedDiagLevel{S, V}, h::UInt) where {S, V} =
    hash(S, hash(V, hash(:ShiftedDiagLevel, h)))

# ─── Level format predicates ──────────────────────────────────────────────────

is_ordered(::AbstractLevelFormat) = true
is_ordered(::CompressedLevel{U,O}) where {U,O} = O

is_unique(::AbstractLevelFormat) = true
is_unique(::CompressedLevel{U,O}) where {U,O} = U

# ─── Dimension ────────────────────────────────────────────────────────────────

struct Dimension
    name::Symbol
end

Base.:(==)(a::Dimension, b::Dimension) = a.name == b.name
Base.hash(d::Dimension, h::UInt)       = hash(d.name, h)
Base.show(io::IO, d::Dimension)        = print(io, d.name)

# ─── LevelExpr ────────────────────────────────────────────────────────────────

struct LevelExpr
    op::Symbol   # :add, :sub, :div, :mod
    lhs::Union{Dimension,LevelExpr}
    rhs::Union{Dimension,LevelExpr,Int}
end

Base.:(==)(a::LevelExpr, b::LevelExpr) =
    a.op == b.op && a.lhs == b.lhs && a.rhs == b.rhs
Base.hash(e::LevelExpr, h::UInt) = hash(e.op, hash(e.lhs, hash(e.rhs, h)))

const _OP_DISPLAY = Dict(:add => "+", :sub => "-", :div => "÷", :mod => "%")

function Base.show(io::IO, e::LevelExpr)
    print(io, "(", e.lhs, " ", _OP_DISPLAY[e.op], " ", e.rhs, ")")
end

# ─── Arithmetic operators on Dimension / LevelExpr ───────────────────────────

const _KeyLike = Union{Dimension,LevelExpr}

Base.:+(a::_KeyLike, b::Union{_KeyLike,Int}) = LevelExpr(:add, a, b)
Base.:-(a::_KeyLike, b::Union{_KeyLike,Int}) = LevelExpr(:sub, a, b)
# Accept any RHS so validation can produce the right error message.
Base.:÷(a::_KeyLike, b::Union{_KeyLike,Int}) = LevelExpr(:div, a, b)
Base.:%(a::_KeyLike, b::Union{_KeyLike,Int}) = LevelExpr(:mod, a, b)

# ─── Convenience constructors ─────────────────────────────────────────────────

dims(names::Symbol...) = Dimension[Dimension(n) for n in names]

# ─── TensorFormat ─────────────────────────────────────────────────────────────

const _LevelKey  = Union{Dimension,LevelExpr}
const _LevelPair = Pair{_LevelKey, AbstractLevelFormat}

# Level structure (the tuple type of level instances) is encoded in the type
# parameter so kernel emitters can dispatch on it directly — no value-level
# format equality, no global Dict cache, no invokelatest.
#
# `keys` (the level expressions) and `levels` (the level instances) used to be
# coupled as a Vector of Pairs.  They are now stored separately: keys remain a
# runtime Vector for dim2lvl/lvl2dim metadata; levels is a typed tuple whose
# type captures the format's structure for dispatch.
# `Family` (Symbol) lifts the format family tag into the type system so capability
# checks (e.g. is-BSR, is-SELL) are pure dispatch.  For non-parametric formats
# Family equals the format's name; parametric builders share a family across all
# parameter values (Formats.BSRRight((2,2)) and Formats.BSRRight((4,4)) both have
# Family = :BSR but distinct LevelTypes).
struct TensorFormat{LevelTypes <: Tuple, Family}
    dimensions  :: Vector{Dimension}
    keys        :: Vector{_LevelKey}     # parallel to `levels`
    levels      :: LevelTypes            # tuple of level instances
    name        :: Symbol
    is_identity :: Bool
    is_ordered  :: Bool
    is_unique   :: Bool
end

# Two TensorFormats are equal when their dimensions, keys, and level types match.
# The level instances themselves are singletons (parametric on their distinguishing
# fields), so equality reduces to type equality on `levels`.
Base.:(==)(a::TensorFormat, b::TensorFormat) =
    a.dimensions == b.dimensions && a.keys == b.keys && typeof(a.levels) === typeof(b.levels)

function Base.hash(f::TensorFormat, h::UInt)
    for d in f.dimensions; h = hash(d, h); end
    for k in f.keys;       h = hash(k, h); end
    for v in f.levels;     h = hash(v, h); end
    h
end

# ─── Internal helpers for dim2lvl / lvl2dim ───────────────────────────────────

@inline _dim_pos(d::Dimension, dims) = findfirst(==(d), dims)::Int

function _evaluate(key::Dimension, dims, dim_indices, _as_size)
    dim_indices[_dim_pos(key, dims)]
end

function _evaluate(key::LevelExpr, dims, dim_indices, as_size)
    e1 = _evaluate(key.lhs, dims, dim_indices, as_size)
    e2 = key.rhs isa Int ? key.rhs : _evaluate(key.rhs, dims, dim_indices, as_size)
    if key.op == :add
        as_size ? e1 + e2 - 1 : e1 + e2
    elseif key.op == :sub
        as_size ? e1 + e2 - 1 : e1 - e2
    elseif key.op == :div
        e1 ÷ e2
    else   # :mod
        as_size ? e2 : e1 % e2
    end
end

function _find_range_level(keys, levels, expr1, expr2)
    for rl in eachindex(keys)
        k = keys[rl]
        v = levels[rl]
        if v isa RangeLevel && (k == expr1 || k == expr2)
            return rl, k
        end
    end
    error("_find_range_level: cannot find range level for $expr1 or $expr2")
end

function _invert!(key::LevelExpr, dims, keys, levels, dim_indices, lvl_indices, idx)
    expr1, expr2 = key.lhs, key.rhs
    if key.op == :add
        rl, re = _find_range_level(keys, levels, expr1, expr2)
        if re == expr2
            dim_indices[_dim_pos(expr1, dims)] = lvl_indices[idx] - lvl_indices[rl]
        else
            dim_indices[_dim_pos(expr2, dims)] = lvl_indices[idx] - lvl_indices[rl]
        end
    elseif key.op == :sub
        rl, re = _find_range_level(keys, levels, expr1, expr2)
        if re == expr2
            dim_indices[_dim_pos(expr1, dims)] = lvl_indices[rl] + lvl_indices[idx]
        else
            dim_indices[_dim_pos(expr2, dims)] = lvl_indices[rl] - lvl_indices[idx]
        end
    elseif key.op == :div
        dim_indices[_dim_pos(expr1, dims)] = lvl_indices[idx] * expr2
    else   # :mod
        dim_indices[_dim_pos(expr1, dims)] += lvl_indices[idx]
    end
end

_invert!(key::Dimension, dims, keys, levels, dim_indices, lvl_indices, idx) =
    (dim_indices[_dim_pos(key, dims)] = lvl_indices[idx])

# ─── Semantic validation ──────────────────────────────────────────────────────

function _semantically_validate(dimensions::Vector{Dimension},
                                 keys::Vector{_LevelKey}, levels::Tuple)
    length(keys) == length(levels) ||
        throw(InvalidTensorFormat("keys and levels have mismatched lengths"))
    all_dims = Set(dimensions)
    length(all_dims) == length(dimensions) ||
        throw(InvalidTensorFormat(
            "Dimension specifications $dimensions has repeated identifiers."))

    used_dims  = Set{Dimension}()
    range_dims = Dict{Dimension, Union{Dimension,Int}}()   # 0 = consumed
    block_dims = Dict{Dimension, Int}()                    # 0 = consumed
    batch      = 0

    for idx in eachindex(keys)
        k = keys[idx]
        v = levels[idx]

        # ── validate level key ──
        if k isa Dimension
            k in all_dims ||
                throw(InvalidTensorFormat("Dimension $k does not appear in $dimensions."))
            push!(used_dims, k)

        elseif k isa LevelExpr
            if k.op == :add || k.op == :sub
                expr1, expr2 = k.lhs, k.rhs
                expr1 in all_dims ||
                    throw(InvalidTensorFormat(
                        "LHS $expr1 in $k does not appear in $dimensions."))
                expr2 in all_dims ||
                    throw(InvalidTensorFormat(
                        "RHS $expr2 in $k does not appear in $dimensions."))
                push!(used_dims, expr1)
                push!(used_dims, expr2)
                (haskey(range_dims, expr1) || haskey(range_dims, expr2)) &&
                    throw(InvalidTensorFormat(
                        "Operation $k reuses dimension of a prior range computation."))
                range_dims[expr1] = expr2
                range_dims[expr2] = expr1

            elseif k.op == :div || k.op == :mod
                expr1, expr2 = k.lhs, k.rhs
                expr1 in all_dims ||
                    throw(InvalidTensorFormat(
                        "LHS $expr1 in $k does not appear in $dimensions."))
                expr2 isa Int ||
                    throw(InvalidTensorFormat(
                        "RHS $expr2 in $k must be an integer."))
                expr2 > 0 ||
                    throw(InvalidTensorFormat(
                        "RHS $expr2 in $k must be strictly positive integer."))
                push!(used_dims, expr1)
                if k.op == :div
                    haskey(block_dims, expr1) &&
                        throw(InvalidTensorFormat(
                            "Division $k reuses dimension of a prior division."))
                    block_dims[expr1] = expr2
                else   # :mod
                    get(block_dims, expr1, 0) != expr2 &&
                        throw(InvalidTensorFormat(
                            "Modulo $k does not match any prior division."))
                    block_dims[expr1] = 0
                end
            else
                throw(InvalidTensorFormat("Unexpected operator in level expression $k."))
            end
        else
            throw(InvalidTensorFormat(
                "Level expression $k must be a Dimension or LevelExpr."))
        end

        # ── validate RangeLevel ──
        if v isa RangeLevel
            k isa Dimension ||
                throw(InvalidTensorFormat("Range uses compound level expression $k."))
            other = get(range_dims, k, 0)
            other == 0 &&
                throw(InvalidTensorFormat(
                    "Range uses dimension $k that is not uniquely defined."))
            range_dims[k]     = 0
            range_dims[other] = 0
        end

        # ── validate batch structure ──
        if v isa BatchLevel
            k == dimensions[idx] ||
                throw(InvalidTensorFormat(
                    "Batch uses non-identity $k for $(dimensions[idx])."))
            batch == -1 &&
                throw(InvalidTensorFormat("Batch is used in inner level $idx."))
            batch += 1
        elseif batch > 0 && !(v isa DenseLevel)
            throw(InvalidTensorFormat("Batch levels must end in a dense level."))
        else
            batch = -1
        end
    end

    # ── post-loop checks ──
    all_dims == used_dims ||
        throw(InvalidTensorFormat(
            "The following dimensions are not used: $(setdiff(all_dims, used_dims))."))

    unmatched_block = [k for (k, v) in block_dims if v != 0]
    isempty(unmatched_block) ||
        throw(InvalidTensorFormat(
            "Some division dimensions are not matched by modulo: $unmatched_block."))

    unmatched_range = [k for (k, v) in range_dims if v != 0]
    isempty(unmatched_range) ||
        throw(InvalidTensorFormat(
            "Some add/sub dimensions are not matched by range: $unmatched_range."))

    batch > 0 &&
        throw(InvalidTensorFormat("Batch levels are not properly closed by a dense level."))
end

# ─── TensorFormat constructor ─────────────────────────────────────────────────

function TensorFormat(
    dimensions,
    level_pairs::AbstractVector;
    name::Symbol   = Symbol(repr(level_pairs)),
    family::Symbol = name,
)
    dims_vec  = convert(Vector{Dimension}, collect(dimensions))
    pairs_vec = Vector{_LevelPair}(collect(level_pairs))
    keys_vec  = _LevelKey[p.first for p in pairs_vec]
    levels    = Tuple(p.second for p in pairs_vec)

    _semantically_validate(dims_vec, keys_vec, levels)

    n = length(dims_vec)
    m = length(keys_vec)

    identity = (n == m) && all(
        begin k = keys_vec[i]; k isa Dimension && _dim_pos(k, dims_vec) == i end
        for i in 1:m
    )
    ordered = all(is_ordered(lv) for lv in levels)
    unique  = isempty(levels) || any(is_unique(lv) for lv in levels)

    LT = typeof(levels)
    TensorFormat{LT, family}(dims_vec, keys_vec, levels, name, identity, ordered, unique)
end

# Scalar (zero-dimensional) convenience
function TensorFormat(::Tuple{}, ::Tuple{}; name::Symbol=:Scalar)
    TensorFormat{Tuple{}, name}(Dimension[], _LevelKey[], (), name, true, true, true)
end

"""
    format_family(fmt) -> Symbol

Return the format family tag.  Non-parametric formats (e.g. `Formats.CSR`) have
`family == name`.  Parametric builders (e.g. `Formats.BSRRight((2,2))`) share a
common family tag (`:BSR`) across all block sizes, enabling family-level capability
queries without string manipulation.
"""
format_family(::TensorFormat{LT, F}) where {LT, F} = F
format_family(::Type{<:TensorFormat{LT, F}}) where {LT, F} = F

# ─── Public API ───────────────────────────────────────────────────────────────

Base.length(fmt::TensorFormat)   = length(fmt.keys)
Base.ndims(fmt::TensorFormat)    = length(fmt.dimensions)

function dim2lvl(fmt::TensorFormat, dim_indices; as_size::Bool=false)
    [_evaluate(k, fmt.dimensions, dim_indices, as_size) for k in fmt.keys]
end

function lvl2dim(fmt::TensorFormat, lvl_indices)
    dim_indices = zeros(Int, length(fmt.dimensions))
    for idx in eachindex(fmt.keys)
        _invert!(fmt.keys[idx], fmt.dimensions, fmt.keys, fmt.levels,
                 dim_indices, lvl_indices, idx)
    end
    dim_indices
end

# ─── Display ──────────────────────────────────────────────────────────────────

Base.show(io::IO, ::DenseLevel{nothing}) = print(io, "DenseLevel")
Base.show(io::IO, ::DenseLevel{Sz}) where {Sz} = print(io, "DenseLevel($Sz)")
Base.show(io::IO, ::BatchLevel{nothing}) = print(io, "BatchLevel")
Base.show(io::IO, ::BatchLevel{Sz}) where {Sz} = print(io, "BatchLevel($Sz)")
Base.show(io::IO, ::SingletonLevel)  = print(io, "SingletonLevel")
Base.show(io::IO, ::RangeLevel)      = print(io, "RangeLevel")
Base.show(io::IO, ::DeltaLevel{B}) where {B} = print(io, "DeltaLevel($B)")
function Base.show(io::IO, ::CompressedLevel{U,O}) where {U,O}
    if U && O
        print(io, "CompressedLevel")
    elseif !U && O
        print(io, "CompressedLevel(nonunique)")
    elseif !U && !O
        print(io, "CompressedLevel(nonunique, unordered)")
    else
        print(io, "CompressedLevel(unordered)")
    end
end

function Base.show(io::IO, fmt::TensorFormat)
    dims_str = join(fmt.dimensions, ", ")
    parts    = ["$(fmt.keys[i]): $(fmt.levels[i])" for i in eachindex(fmt.keys)]
    print(io, "[", dims_str, "] -> (", join(parts, ", "), ")")
end

# ─── @tensor_format macro ─────────────────────────────────────────────────────

# These helpers run at macro-expansion time on raw AST nodes.

function _tf_build_key(node)
    if node isa Symbol
        :(Dimension($(QuoteNode(node))))
    elseif node isa Expr && node.head == :call
        op_sym = Dict(:- => :sub, :+ => :add, :÷ => :div, :% => :mod,
                      :div => :div)[node.args[1]]
        lhs = _tf_build_key(node.args[2])
        rhs_raw = node.args[3]
        rhs = rhs_raw isa Integer ? rhs_raw :
              rhs_raw isa Symbol  ? :(Dimension($(QuoteNode(rhs_raw)))) :
                                    _tf_build_key(rhs_raw)
        :(LevelExpr($(QuoteNode(op_sym)), $lhs, $rhs))
    else
        error("@tensor_format: cannot parse level key: $node")
    end
end

# Static size from a level key expression: `_ % N` and `_ ÷ N` patterns paired
# with `dense` get sized DenseLevel(N) so blocked-modulo dims (BSR/SELL/etc)
# carry their compile-time extent in the type.  Returns Int or nothing.
function _tf_static_size(node)
    (node isa Expr && node.head == :call && length(node.args) == 3) || return nothing
    node.args[1] == :%             || return nothing
    node.args[3] isa Integer       || return nothing
    Int(node.args[3])
end

function _tf_build_fmt(node, static_size::Union{Int,Nothing}=nothing)
    if node isa Symbol
        if node == :dense
            static_size === nothing ? :(DenseLevel()) : :(DenseLevel($static_size))
        elseif node == :batch
            :(BatchLevel())
        elseif node == :compressed
            :(CompressedLevel())
        elseif node == :singleton
            :(SingletonLevel())
        elseif node == :range
            :(RangeLevel())
        else
            error("@tensor_format: unknown level format symbol: $node")
        end
    elseif node isa Expr && node.head == :call
        fname, fargs = node.args[1], node.args[2:end]
        if fname == :compressed
            uniq = :nonunique ∉ fargs
            ord  = :unordered ∉ fargs
            :(CompressedLevel(unique=$uniq, ordered=$ord))
        elseif fname == :delta
            :(DeltaLevel($(fargs[1])))
        else
            error("@tensor_format: unknown level format call: $fname")
        end
    else
        error("@tensor_format: cannot parse level format: $node")
    end
end

function _tf_build_pair(spec)
    (spec isa Expr && spec.head == :call && spec.args[1] == :(:)) ||
        error("@tensor_format: expected `key : format`, got: $spec")
    key_node    = spec.args[2]
    static_size = _tf_static_size(key_node)
    key_expr    = _tf_build_key(key_node)
    fmt_expr    = _tf_build_fmt(spec.args[3], static_size)
    :($key_expr => $fmt_expr)
end

macro tensor_format(name, arrow_expr)
    (arrow_expr isa Expr && arrow_expr.head == :->) ||
        error("@tensor_format: expected `(dims) -> (levels)`, got: $arrow_expr")

    dims_node = arrow_expr.args[1]
    raw_body  = arrow_expr.args[2]

    # Julia wraps single-expression lambda bodies in a :block node with a LineNumberNode.
    lvls_node = if raw_body isa Expr && raw_body.head == :block
        actual = filter(x -> !(x isa LineNumberNode), raw_body.args)
        length(actual) == 1 ? actual[1] : raw_body
    else
        raw_body
    end

    dim_syms = if dims_node isa Symbol
        [dims_node]
    elseif dims_node isa Expr && dims_node.head == :tuple
        dims_node.args
    else
        error("@tensor_format: expected dimension tuple, got: $dims_node")
    end

    lvl_specs = if lvls_node isa Expr && lvls_node.head == :tuple
        lvls_node.args
    elseif lvls_node isa Expr && lvls_node.head == :call && lvls_node.args[1] == :(:)
        [lvls_node]
    else
        error("@tensor_format: expected level spec tuple, got: $lvls_node")
    end

    dim_exprs  = [:(Dimension($(QuoteNode(s)))) for s in dim_syms]
    pair_exprs = [_tf_build_pair(s) for s in lvl_specs]

    # No `const` — works at any scope (module level or inside a testset/function).
    # At module scope the binding is effectively constant; inside a scope it's a local.
    # family defaults to name so non-parametric formats are self-identifying.
    quote
        $(esc(name)) = TensorFormat(
            [$(dim_exprs...)],
            [$(pair_exprs...)];
            name   = $(QuoteNode(name)),
            family = $(QuoteNode(name)),
        )
    end
end
