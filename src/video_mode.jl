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

    CImGui.Bullet()
    CImGui.TextWrapped("Capturing with the same render settings as in the main screen.")

    CImGui.Bullet()
    CImGui.TextWrapped("Window is not resizable during this stage.")

    CImGui.Text("N Keyframes: $(length(video_mode.camera_path))")

    if video_mode.is_rendering
        CImGui.TextWrapped("Capturing. Please wait...")

        n_steps = video_mode.steps_ref[] * (length(video_mode.camera_path) - 1)
        current_step = video_mode.camera_path.current_step

        CImGui.PushStyleColor(
            CImGui.ImGuiCol_PlotHistogram, CImGui.HSV(0.61f0, 1.0f0, 1f0))
        CImGui.ProgressBar(
            current_step / (n_steps + 1), CImGui.ImVec2(-1f0, 0f0),
            "$current_step / $(n_steps + 1)")
        CImGui.PopStyleColor()
    else
        if CImGui.CollapsingHeader("Video Settings", CIM_HEADER)
            CImGui.PushItemWidth(-100)
            CImGui.SliderInt(
                "Lerp Steps", video_mode.steps_ref, 1, 60, "%d / 60")
            CImGui.PushItemWidth(-100)
            CImGui.SliderInt(
                "Frame rate", video_mode.framerate_ref, 1, 60, "%d")
            CImGui.PushItemWidth(-100)
            CImGui.Checkbox("Save frames", video_mode.save_frames)

            CImGui.PushItemWidth(-100)
            CImGui.InputText(
                "Save Directory", pointer(video_mode.save_dir),
                length(video_mode.save_dir))
        end

        # Select camera path mode.
        if CImGui.CollapsingHeader("Camera Path Settings", CIM_HEADER)
            CImGui.TextWrapped(
                "Press 'V' to add camera position (at least 2 are required).")

            red_button_begin()
            if CImGui.Button("Clear Path")
                empty!(video_mode.camera_path)
                ngui.render_state.need_render = true
            end
            red_button_end()
        end
    end

    if length(video_mode.camera_path) ≥ 2
        CImGui.SameLine()
        if CImGui.Button("Capture")
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
    end

    if CImGui.Button("Go Back")
        ngui.screen = MainScreen
        video_mode.is_rendering = false
        close_video!(video_mode)
        GL.set_resizable_window!(ngui.context, true)
    end
    CImGui.End()
end

function loop!(video_mode::VideoMode; ngui)
    frame_time = update_time!(ngui.render_state)

    GL.imgui_begin(ngui.context)
    handle_ui!(video_mode; ngui)

    if !video_mode.is_rendering && !is_mouse_in_ui()
        ngui.render_state.need_render |= handle_keyboard!(
            ngui.controls, ngui.renderer.camera; frame_time)
        ngui.render_state.need_render |= handle_mouse!(
            ngui.controls, ngui.renderer.camera)

        if GL.is_key_pressed('V'; repeat=false)
            push!(video_mode.camera_path, ngui.renderer.camera)
            ngui.render_state.need_render = true
        end
    end

    GL.clear()
    GL.set_clear_color(0.2, 0.2, 0.2, 1.0)

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
    GL.draw(ngui.render_state.surface)

    GL.clear(GL.GL_DEPTH_BUFFER_BIT)

    if !video_mode.is_rendering
        P = GL.perspective(ngui.renderer.camera)
        L = GL.look_at(ngui.renderer.camera)
        GL.draw(video_mode.camera_path, P, L)
    end

    GL.imgui_end(ngui.context)
    glfwSwapBuffers(ngui.context.window)
    glfwPollEvents()
    return nothing
end
