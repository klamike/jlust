struct InvalidTensorFormat  <: Exception; msg::String end
struct InvalidLevelAccess   <: Exception; msg::String end
struct UnsupportedFormat    <: Exception; format::Any; backend::Any end
struct UnsupportedIndexType <: Exception; T::Type; backend::Any end
struct UnsupportedValueType <: Exception; T::Type; backend::Any end
struct IndexOriginMismatch  <: Exception; have::Any; need::Any end
struct DeviceMismatch       <: Exception; msg::String end
struct IncompatibleExtents  <: Exception; msg::String end
struct NonCanonicalStorage  <: Exception; msg::String end

Base.showerror(io::IO, e::InvalidTensorFormat) = print(io, "InvalidTensorFormat: ", e.msg)
Base.showerror(io::IO, e::InvalidLevelAccess)  = print(io, "InvalidLevelAccess: ", e.msg)
Base.showerror(io::IO, e::DeviceMismatch)      = print(io, "DeviceMismatch: ", e.msg)
Base.showerror(io::IO, e::IncompatibleExtents) = print(io, "IncompatibleExtents: ", e.msg)
Base.showerror(io::IO, e::NonCanonicalStorage) = print(io, "NonCanonicalStorage: ", e.msg)
Base.showerror(io::IO, e::IndexOriginMismatch) =
    print(io, "IndexOriginMismatch: have $(e.have), need $(e.need)")
Base.showerror(io::IO, e::UnsupportedFormat) =
    print(io, "UnsupportedFormat: $(e.format) not supported by $(e.backend)")
Base.showerror(io::IO, e::UnsupportedIndexType) =
    print(io, "UnsupportedIndexType: $(e.T) not supported by $(e.backend)")
Base.showerror(io::IO, e::UnsupportedValueType) =
    print(io, "UnsupportedValueType: $(e.T) not supported by $(e.backend)")
