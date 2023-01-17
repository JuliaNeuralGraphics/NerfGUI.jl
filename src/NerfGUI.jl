module NerfGUI

using CImGui
using CImGui.ImGuiGLFWBackend.LibGLFW
using FileIO
using GL
using ImageCore
using ImageTransformations
using LinearAlgebra
using ModernGL
using Nerf
using Preferences
using Rotations
using StaticArrays
using VideoIO

const UUID = Base.UUID("4cbd8c4d-76eb-460a-a95f-3d783f5c44b5")
const DEVICE = Nerf.DEVICE

const CIM_HEADER =
    CImGui.ImGuiTreeNodeFlags_CollapsingHeader |
    CImGui.ImGuiTreeNodeFlags_DefaultOpen

function is_mouse_in_ui()
    CImGui.IsMousePosValid() && unsafe_load(CImGui.GetIO().WantCaptureMouse)
end

# Extend GL a bit.
function GL.look_at(c::Nerf.Camera)
    GL.look_at(Nerf.view_pos(c), Nerf.look_at(c), -Nerf.view_up(c))
end

function GL.perspective(c::Nerf.Camera; near::Float32 = 0.1f0, far::Float32 = 100f0)
    fovy = Nerf.focal2fov(c.intrinsics.resolution[2], c.intrinsics.focal[2])
    aspect = c.intrinsics.resolution[1] / c.intrinsics.resolution[2]
    GL.perspective(fovy, aspect, near, far)
end

include("ui_state.jl")
include("render_state.jl")
include("occupancy.jl")
include("camera_path.jl")
include("video_mode.jl")
include("meshing_mode.jl")

@enum Screen begin
    MainScreen
    CaptureScreen
    MeshingScreen
end

mutable struct NGUI
    context::GL.Context
    trainer::Nerf.Trainer
    renderer::Nerf.Renderer
    screen::Screen

    video_mode::VideoMode
    meshing_mode::MeshingMode

    ui_state::UIState
    render_state::RenderState
    controls::ControlSettings

    bbox::GL.BBox
    frustum::GL.Frustum
    occupancy::OccupancyView
end

const NGUI_REF::Ref{NGUI} = Ref{NGUI}()

function resize_callback(_, width, height)
    (width == 0 || height == 0) && return nothing # Window minimized.
    GL.set_viewport(width, height)

    isassigned(NGUI_REF) || return nothing

    ngui::NGUI = NGUI_REF[]
    update_resolutions!(ngui.ui_state; width, height)
    resolution = get_resolution(ngui.ui_state)

    GL.resize!(ngui.render_state.surface; resolution...)
    Nerf.resize!(ngui.renderer; resolution...)
    ngui.render_state.need_render = true
    nothing
end

function NGUI(; gl_kwargs...)
    dev = DEVICE

    GL.init(3, 0)
    context = GL.Context("NerfGUI"; gl_kwargs...)
    GL.set_resize_callback!(context, resize_callback)

    datasets = String[
        joinpath(pkgdir(Nerf), "data", "raccoon_sofa2", "transforms.json"),
    ]

    ui_state = UIState(datasets; width=context.width, height=context.height)
    resolution = get_resolution(ui_state)
    render_state = RenderState(; surface=GL.RenderSurface(; resolution...))
    controls = ControlSettings()

    n_rays = load_preference(UUID, "n_rays")
    tile_size = load_preference(UUID, "tile_size")

    dataset = Nerf.Dataset(dev; config_file=datasets[1])
    camera = Nerf.Camera(MMatrix{3, 4, Float32}(I), dataset.intrinsics)
    Nerf.set_projection!(camera, Nerf.get_pose(dataset, 1)...)
    Nerf.set_resolution!(camera; resolution...)

    model = Nerf.BasicModel(Nerf.BasicField(dev))
    trainer = Nerf.Trainer(model, dataset; n_rays)
    renderer = Nerf.Renderer(dev, camera, trainer.bbox, trainer.cone; tile_size)

    ui_state.bbox_min = Vector(renderer.bbox.min)
    ui_state.bbox_max = Vector(renderer.bbox.max)
    bbox = GL.BBox(renderer.bbox.min, renderer.bbox.max)

    ngui = NGUI(
        context, trainer, renderer, MainScreen,
        VideoMode(), MeshingMode(),
        ui_state, render_state, controls,
        bbox, GL.Frustum(), OccupancyView())
    NGUI_REF[] = ngui
    ngui
end

function reset!(ngui::NGUI)
    Nerf.reset!(ngui.trainer)
    Nerf.reset!(ngui.renderer)
    reset_ui!(ngui)
    ngui.renderer.mode = Nerf.Color # Default mode is color.
    ngui.render_state.need_render = true
    return nothing
end

function reset_ui!(ngui::NGUI)
    ngui.ui_state = UIState(ngui.ui_state)
    ngui.ui_state.bbox_min = Vector(ngui.renderer.bbox.min)
    ngui.ui_state.bbox_max = Vector(ngui.renderer.bbox.max)

    new_resolution = get_resolution(ngui.ui_state)
    GL.resize!(ngui.render_state.surface; new_resolution...)
    Nerf.resize!(ngui.renderer; new_resolution...)
    return nothing
end

function change_dataset!(ngui::NGUI)
    dev = Nerf.get_device(ngui.renderer)
    config_file = ngui.ui_state.datasets[ngui.ui_state.dataset_idx[] + 1]
    # TODO free the old dataset before creating a new one.
    dataset = Nerf.Dataset(dev; config_file)

    Nerf.set_dataset!(ngui.trainer, dataset)
    Nerf.set_dataset!(ngui.renderer, dataset, ngui.trainer.cone, ngui.trainer.bbox)
    Nerf.set_projection!(ngui.renderer.camera, Nerf.get_pose(dataset, 1)...)
    reset_ui!(ngui)
    return nothing
end

function render!(ngui::NGUI)::Nothing
    # Do not render if we completely finished rendering the frame.
    !ngui.render_state.need_render &&
        ngui.render_state.finished_frame && return nothing

    # `need_render` is `true` every time user interacts with the app
    # via controls, so we need to render anew.
    if ngui.render_state.need_render
        Nerf.reset!(ngui.renderer)
        ngui.render_state.finished_frame = false
        ngui.render_state.need_render = false
    end

    ngui.render_state.finished_frame && return nothing

    Nerf.render_tile!(
        ngui.renderer, ngui.trainer.occupancy, ngui.trainer.bbox;
        normals_consumer=(points) -> Nerf.∇normals(ngui.trainer.model, points),
    ) do points, directions
        ngui.trainer.model(points, directions)
    end
    Nerf.advance_tile!(ngui.renderer)

    if Nerf.is_done(ngui.renderer) # If finished all tiles, increase spp.
        ngui.renderer.buffer.spp += 1
        ngui.render_state.finished_frame =
            ngui.renderer.buffer.spp == ngui.ui_state.spp[]

        # If spp < than required, reset tile counter and clear buffer.
        if !ngui.render_state.finished_frame
            Nerf.clear!(ngui.renderer.buffer)
            ngui.renderer.tile_idx = 0
        end
    end

    img = Nerf.to_gl_texture(ngui.renderer.buffer)
    GL.set_data!(ngui.render_state.surface, img)
    return nothing
end

function launch!(ngui::NGUI)
    GL.render_loop(ngui.context) do
        if ngui.screen == MainScreen
            loop!(ngui)
        elseif ngui.screen == CaptureScreen
            loop!(ngui.video_mode; ngui)
        elseif ngui.screen == MeshingScreen
            loop!(ngui.meshing_mode; ngui)
        end
        return true
    end
end

function handle_ui!(ngui::NGUI; frame_time)
    CImGui.Begin("NeRF")

    # Draw FPS info.
    frame_time_ms = ceil(Int, frame_time * 1000)
    fps = ceil(Int, 1f0 / (frame_time + 1f-6))
    CImGui.Text("Frame: $frame_time_ms ms ($fps FPS);")

    # Change dataset UI.
    CImGui.PushItemWidth(-100)
    if CImGui.Combo(
        "[$(length(ngui.ui_state.datasets))] Scenes", ngui.ui_state.dataset_idx,
        ngui.ui_state.datasets, length(ngui.ui_state.datasets),
    )
        change_dataset!(ngui)
        GL.update_corners!(ngui.bbox,
            ngui.renderer.bbox.min, ngui.renderer.bbox.max)
        ngui.render_state.need_render = true
    end

    CImGui.PushItemWidth(-1)
    CImGui.InputText(
        "", pointer(ngui.ui_state.transforms_file),
        length(ngui.ui_state.transforms_file))

    if CImGui.Button("Add Dataset")
        transforms_file = unsafe_string(pointer(ngui.ui_state.transforms_file))
        if isfile(transforms_file)
            add_dataset!(ngui.ui_state, transforms_file)
            ngui.ui_state.transforms_file = Vector{UInt8}(
                "Path to 'transforms.json'" * "\0"^512)
        else
            ngui.ui_state.transforms_file = Vector{UInt8}(
                "[Error] Need correct path to 'transforms.json'" * "\0"^512)
        end
    end

    if CImGui.CollapsingHeader("Training", CIM_HEADER)
        CImGui.Checkbox("Train", ngui.ui_state.train)

        mean_density = ngui.trainer.occupancy.mean_density
        CImGui.Text("Steps: $(ngui.trainer.step) Loss: $(ngui.ui_state.loss)")
        CImGui.Text("Mean density: $mean_density ($(log(mean_density)) log)")
        CImGui.PushItemWidth(-1)
        CImGui.PlotLines(
            "", ngui.ui_state.loss_history, length(ngui.ui_state.loss_history),
            0, "Loss", CImGui.FLT_MAX, CImGui.FLT_MAX,
            CImGui.ImVec2(0f0, 50f0))
    end

    if CImGui.CollapsingHeader("Rendering", CIM_HEADER)
        CImGui.Checkbox("Render", ngui.ui_state.render)
        CImGui.Checkbox("Show render bbox", ngui.ui_state.draw_bbox)
        CImGui.Checkbox("Show camera poses", ngui.ui_state.draw_poses)
        CImGui.Checkbox("Show occupancy", ngui.ui_state.draw_occupancy)

        max_level = Nerf.get_n_levels(ngui.trainer.dataset)
        CImGui.PushItemWidth(-150)
        if CImGui.SliderInt(
            "Max occupancy level", ngui.ui_state.occupancy_n_levels,
            0, max_level, "%d / $max_level",
        )
            update!(ngui.occupancy, ngui.trainer.occupancy;
                n_levels=ngui.ui_state.occupancy_n_levels[],
                update_step=ngui.trainer.step)
        end

        # Render bbox ⊆ training bbox.
        bbox_min_v = minimum(ngui.trainer.bbox.min)
        bbox_max_v = maximum(ngui.trainer.bbox.max)
        CImGui.PushItemWidth(-150)
        if CImGui.SliderFloat3(
            "Render bbox min", pointer(ngui.ui_state.bbox_min),
            bbox_min_v, bbox_max_v,
        )
            new_min = SVector{3, Float32}(ngui.ui_state.bbox_min...)
            train_min = ngui.trainer.bbox.min
            train_max = ngui.trainer.bbox.max
            old_bbox = ngui.renderer.bbox

            ngui.renderer.bbox = BBox(
                max.(min.(new_min, old_bbox.max), train_min),
                min.(max.(new_min, old_bbox.max), train_max))
            GL.update_corners!(ngui.bbox,
                ngui.renderer.bbox.min, ngui.renderer.bbox.max)
            ngui.render_state.need_render = true
        end

        CImGui.PushItemWidth(-150)
        if CImGui.SliderFloat3(
            "Render bbox max", pointer(ngui.ui_state.bbox_max),
            bbox_min_v, bbox_max_v,
        )
            new_max = SVector{3, Float32}(ngui.ui_state.bbox_max...)
            train_min = ngui.trainer.bbox.min
            train_max = ngui.trainer.bbox.max
            old_bbox = ngui.renderer.bbox

            ngui.renderer.bbox = BBox(
                max.(min.(new_max, old_bbox.min), train_min),
                min.(max.(new_max, old_bbox.min), train_max))
            GL.update_corners!(ngui.bbox,
                ngui.renderer.bbox.min, ngui.renderer.bbox.max)
            ngui.render_state.need_render = true
        end

        CImGui.PushItemWidth(-100)
        if CImGui.Combo(
            "Resolution", ngui.ui_state.resolution_idx,
            ngui.ui_state.resolution_labels, length(ngui.ui_state.resolutions),
        )
            new_resolution = get_resolution(ngui.ui_state)
            GL.resize!(ngui.render_state.surface; new_resolution...)
            Nerf.resize!(ngui.renderer; new_resolution...)
            ngui.render_state.need_render = true
        end

        CImGui.PushItemWidth(-100)
        if CImGui.SliderInt("Samples/pixel", ngui.ui_state.spp, 1, 8, "%d / 8")
            ngui.render_state.need_render = true
        end

        CImGui.PushItemWidth(-100)
        if CImGui.Combo(
            "Render mode", ngui.ui_state.render_mode_idx,
            ngui.ui_state.render_modes, length(ngui.ui_state.render_modes),
        )
            ngui.renderer.mode = Nerf.RenderMode(ngui.ui_state.render_mode_idx[])
            ngui.render_state.need_render = true
        end

        frame_filenames = ngui.trainer.dataset.frame_filenames
        CImGui.PushItemWidth(-100)
        if CImGui.Combo(
            "View", ngui.ui_state.selected_view,
            frame_filenames, length(frame_filenames),
        )
            vid = ngui.ui_state.selected_view[] + 1
            Nerf.set_projection!(
                ngui.renderer.camera,
                Nerf.get_pose(ngui.trainer.dataset, vid)...)
            ngui.render_state.need_render = true
        end
    end

    if CImGui.Button("Capture Mode")
        ngui.screen = CaptureScreen
        GL.set_resizable_window!(ngui.context, false)
    end
    if CImGui.Button("Meshing Mode")
        ngui.screen = MeshingScreen
    end

    red_button_begin()
    CImGui.Button("Reset") && reset!(ngui)
    red_button_end()

    CImGui.End()
    return nothing
end

function loop!(ngui::NGUI)
    frame_time = update_time!(ngui.render_state)

    GL.imgui_begin(ngui.context)
    handle_ui!(ngui; frame_time)
    ngui.render_state.need_render |= handle_keyboard!(
        ngui.controls, ngui.renderer.camera; frame_time)
    ngui.render_state.need_render |= handle_mouse!(
        ngui.controls, ngui.renderer.camera)

    do_train = ngui.ui_state.train[] && !is_mouse_in_ui()
    if do_train
        loss = 0f0
        for _ in 1:16 # TODO Allow configuring.
            loss += Nerf.step!(ngui.trainer)
        end
        ngui.ui_state.loss = loss / 16f0
        push!(ngui.ui_state.loss_history, ngui.ui_state.loss)
        ngui.render_state.need_render = true
    end

    GL.clear()
    GL.set_clear_color(0.2, 0.2, 0.2, 1.0)
    if ngui.ui_state.render[]
        render!(ngui)
        GL.draw(ngui.render_state.surface)
    end
    GL.clear(GL.GL_DEPTH_BUFFER_BIT)

    P = GL.perspective(ngui.renderer.camera)
    L = GL.look_at(ngui.renderer.camera)
    ngui.ui_state.draw_bbox[] && GL.draw(ngui.bbox, P, L)

    if ngui.ui_state.draw_occupancy[]
        step = ngui.trainer.step
        if step % 10 == 0
            update!(ngui.occupancy, ngui.trainer.occupancy;
                n_levels=ngui.ui_state.occupancy_n_levels[], update_step=step)
        end
        GL.draw(ngui.occupancy, P, L)
    end

    if ngui.ui_state.draw_poses[]
        dataset = ngui.trainer.dataset
        for view_id in 1:length(dataset)
            camera = Nerf.get_pose_camera(dataset, view_id)
            camera_perspective =
                GL.perspective(camera; near=0.1f0, far=0.2f0) *
                GL.look_at(camera)
            GL.draw(ngui.frustum, camera_perspective, P, L)
        end
    end

    GL.imgui_end(ngui.context)
    glfwSwapBuffers(ngui.context.window)
    glfwPollEvents()
    return nothing
end

function main()
    if load_preference(UUID, "fullscreen")
        ngui = NGUI(; fullscreen=true, resizable=false)
    else
        ngui = NGUI(; width=1920, height=1080, resizable=true)
    end
    launch!(ngui)
end

end
