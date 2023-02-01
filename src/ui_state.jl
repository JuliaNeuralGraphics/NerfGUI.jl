Base.@kwdef mutable struct UIState
    datasets::Vector{String}
    dataset_idx::Ref{Int32} = Ref{Int32}(0)

    render_modes::Vector{String}
    render_mode_idx::Ref{Int32} = Ref{Int32}(4) # Color mode is default.

    resolutions::Vector{Tuple{Int64, Int64}}
    resolution_labels::Vector{String}
    resolution_idx::Ref{Int32} = Ref{Int32}(0)

    transforms_file::Vector{UInt8} = Vector{UInt8}(
        "Specify path to 'transforms.json' file" * "\0"^512)

    train::Ref{Bool} = Ref(load_preference(UUID, "train", true))
    render::Ref{Bool} = Ref(true)

    draw_bbox::Ref{Bool} = Ref(false)
    draw_poses::Ref{Bool} = Ref(false)
    selected_view::Ref{Int32} = Ref{Int32}(0)

    draw_occupancy::Ref{Bool} = Ref(false)
    occupancy_n_levels::Ref{Int32} = Ref{Int32}(0)

    bbox_min::Vector{Float32} = zeros(Float32, 3)
    bbox_max::Vector{Float32} = ones(Float32, 3)

    loss::Float32 = 0f0 # most recent calculated loss
    loss_history::Vector{Float32} = Float32[]

    spp::Ref{Int32} = Ref{Int32}(1)
end

function UIState(datasets::Vector{String}; width::Integer, height::Integer)
    render_modes = string.(collect(instances(Nerf.RenderMode)))
    resolutions, resolution_labels = get_resolutions(; width, height)
    UIState(; datasets, render_modes, resolutions, resolution_labels)
end

function UIState(s::UIState)
    UIState(;
        bbox_min=s.bbox_min, bbox_max=s.bbox_max,
        render_modes=s.render_modes, resolutions=s.resolutions,
        resolution_labels=s.resolution_labels, datasets=s.datasets)
end

function add_dataset!(s::UIState, dataset::String)
    push!(s.datasets, dataset)
end

function get_resolutions(; width::Integer, height::Integer)
    w, n_resolutions = 128, 5
    aspect = height / width

    resolutions = Tuple{Int64, Int64}[]
    push!(resolutions, (w, ceil(Int64, w * aspect))) # TODO round to 2 or 4 as ffmpeg requires it

    w *= 2
    while w < width && length(resolutions) < n_resolutions
        push!(resolutions, (w, ceil(Int64, w * aspect)))
        w *= 2
    end
    push!(resolutions, (width, height))

    resolution_labels = ["$(r[1])x$(r[2])" for r in resolutions]
    resolutions, resolution_labels
end

function get_resolution(s::UIState)
    width, height = s.resolutions[s.resolution_idx[] + 1]
    (; width, height)
end

function update_resolutions!(s::UIState; width::Integer, height::Integer)
    s.resolutions, s.resolution_labels = get_resolutions(; width, height)
    if (s.resolution_idx[] + 1) > length(s.resolutions)
        s.resolution_idx = Ref{Int32}(length(s.resolutions) - 1)
    end
end

function red_button_begin()
    CImGui.PushStyleColor(CImGui.ImGuiCol_Button, CImGui.HSV(0f0, 0.6f0, 0.6f0))
    CImGui.PushStyleColor(CImGui.ImGuiCol_ButtonHovered, CImGui.HSV(0f0, 0.7f0, 0.7f0))
    CImGui.PushStyleColor(CImGui.ImGuiCol_ButtonActive, CImGui.HSV(0f0, 0.7f0, 0.7f0))
end

function red_button_end()
    CImGui.PopStyleColor(3)
end
