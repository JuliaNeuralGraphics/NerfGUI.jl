Base.@kwdef mutable struct VideoMode
    camera_path::CameraPath = CameraPath()
    writer::Union{VideoIO.VideoWriter, Nothing} = nothing
    is_rendering::Bool = false

    # UI stuff.
    save_frames::Ref{Bool} = Ref(false)
    steps_ref::Ref{Int32} = Ref{Int32}(24)
    framerate_ref::Ref{Int32} = Ref{Int32}(24)
    save_dir::Vector{UInt8} = Vector{UInt8}("./" * "\0"^254)
end

function advance!(v::VideoMode, ngui)
    if is_done(v.camera_path)
        v.is_rendering = false
        v.writer ≢ nothing && close_video_out!(v.writer)
    else
        k = eval(v.camera_path)
        Nerf.set_projection!(ngui.renderer.camera, Nerf.get_rotation(k), k.t)
    end
    advance!(v.camera_path, get_time_step(v.camera_path, v.steps_ref[]))
end

function close_video!(v::VideoMode)
    if v.writer ≢ nothing
        close_video_out!(v.writer)
        v.writer = nothing
    end
end

function reset!(v::VideoMode)
    close_video!(v)
    empty!(v.camera_path)
    v.save_dir = Vector{UInt8}("./" * "\0"^254)
end

function handle_ui!(video_mode::VideoMode; ngui)
    CImGui.Begin("Capture Mode")
    CImGui.TextWrapped("Capturing with the same render settings as in the main screen.")
    CImGui.TextWrapped("Window is not resizable during this stage.")
    CImGui.TextWrapped("Press V to add camera position (at least 2 are required).")
    CImGui.Text("N Keyframes: $(length(video_mode.camera_path))")

    if video_mode.is_rendering
        CImGui.TextWrapped("Capturing. Please wait...")

        n_steps = video_mode.steps_ref[] * (length(video_mode.camera_path) - 1)
        current_step = video_mode.camera_path.current_step

        CImGui.PushStyleColor(CImGui.ImGuiCol_PlotHistogram, CImGui.HSV(0.61f0, 1.0f0, 1f0))
        CImGui.ProgressBar(current_step / (n_steps + 1), CImGui.ImVec2(-1f0, 0f0),
            "$current_step / $(n_steps + 1)")
        CImGui.PopStyleColor()

        if CImGui.Button("Cancel", CImGui.ImVec2(-1, 0))
            video_mode.is_rendering = false
            close_video!(video_mode)
        end
    else
        if CImGui.CollapsingHeader("Video Settings", CIM_HEADER)
            CImGui.PushItemWidth(-100)
            CImGui.SliderInt("Lerp Steps", video_mode.steps_ref, 1, 60, "%d / 60")
            CImGui.PushItemWidth(-100)
            CImGui.SliderInt("Frame rate", video_mode.framerate_ref, 1, 60, "%d")

            CImGui.PushItemWidth(-100)
            CImGui.InputText("Save Directory", pointer(video_mode.save_dir),
                length(video_mode.save_dir))

            CImGui.PushItemWidth(-100)
            CImGui.Checkbox("Save frames", video_mode.save_frames)
        end

        CImGui.BeginTable("##capture-buttons-table", 3)
        CImGui.TableNextRow()
        CImGui.TableNextColumn()

        can_capture = length(video_mode.camera_path) ≥ 2
        can_capture || disabled_begin()
        if CImGui.Button("Capture", CImGui.ImVec2(-1, 0))
            reset_time!(video_mode.camera_path)
            close_video!(video_mode)

            # Create directories for video & images.
            save_dir = unsafe_string(pointer(video_mode.save_dir))
            isdir(save_dir) || mkdir(save_dir)
            if video_mode.save_frames[]
                images_dir = joinpath(save_dir, "images")
                isdir(images_dir) || mkdir(images_dir)
            end

            # Open video writer stream.
            video_file = joinpath(save_dir, "out.mp4")
            res = Nerf.get_resolution(ngui.renderer.camera)
            video_mode.writer = open_video_out(
                video_file, zeros(RGB{N0f8}, res[2], res[1]);
                framerate=video_mode.framerate_ref[],
                target_pix_fmt=VideoIO.AV_PIX_FMT_YUV420P)

            video_mode.is_rendering = true
        end
        can_capture || disabled_end()

        CImGui.TableNextColumn()
        if CImGui.Button("Go Back", CImGui.ImVec2(-1, 0))
            ngui.screen = MainScreen
            video_mode.is_rendering = false
            close_video!(video_mode)
            NeuralGraphicsGL.set_resizable_window!(ngui.context, true)
        end

        CImGui.TableNextColumn()
        red_button_begin()
        if CImGui.Button("Clear Path", CImGui.ImVec2(-1, 0))
            empty!(video_mode.camera_path)
            ngui.render_state.need_render = true
        end
        red_button_end()
        CImGui.EndTable()
    end
    CImGui.End()
end

function loop!(video_mode::VideoMode; ngui)
    frame_time = update_time!(ngui.render_state)

    NeuralGraphicsGL.imgui_begin(ngui.context)
    handle_ui!(video_mode; ngui)

    if !video_mode.is_rendering && !is_mouse_in_ui()
        ngui.render_state.need_render |= handle_keyboard!(
            ngui.controls, ngui.renderer.camera; frame_time)
        ngui.render_state.need_render |= handle_mouse!(
            ngui.controls, ngui.renderer.camera)

        if NeuralGraphicsGL.is_key_pressed('V'; repeat=false)
            push!(video_mode.camera_path, ngui.renderer.camera)
            ngui.render_state.need_render = true
        end
    end

    NeuralGraphicsGL.clear()
    NeuralGraphicsGL.set_clear_color(0.2, 0.2, 0.2, 1.0)

    if video_mode.is_rendering
        # Initial advance.
        if video_mode.camera_path.current_step == 0
            advance!(video_mode, ngui)
            ngui.render_state.need_render = true
        end

        if ngui.render_state.finished_frame
            frame = RGB{N0f8}.(Nerf.to_image(ngui.renderer.buffer))

            save_dir = unsafe_string(pointer(video_mode.save_dir))
            if video_mode.save_frames[]
                images_dir = joinpath(save_dir, "images")
                image_file = "nerf-$(video_mode.camera_path.current_step).png"
                save(joinpath(images_dir, image_file), frame)
            end
            write(video_mode.writer, frame)
        end

        if ngui.render_state.finished_frame
            advance!(video_mode, ngui)
            video_mode.is_rendering && (ngui.render_state.need_render = true;)
        end
    end
    render!(ngui)
    NeuralGraphicsGL.draw(ngui.render_state.surface)

    NeuralGraphicsGL.clear(NeuralGraphicsGL.GL_DEPTH_BUFFER_BIT)

    if !video_mode.is_rendering
        P = NeuralGraphicsGL.perspective(ngui.renderer.camera)
        L = NeuralGraphicsGL.look_at(ngui.renderer.camera)
        NeuralGraphicsGL.draw(video_mode.camera_path, P, L)
    end

    NeuralGraphicsGL.imgui_end(ngui.context)
    glfwSwapBuffers(ngui.context.window)
    glfwPollEvents()
    return nothing
end
