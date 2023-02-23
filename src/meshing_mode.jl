Base.@kwdef mutable struct MeshingMode
    algorithm::Ref{Int32} = Ref{Int32}(0)
    algorithm_labels::Vector{String} = ["Marching cubes", "Marching tetrahedra"]

    mc_resolution::Ref{Int32} = Ref{Int32}(0)
    mc_resolutions::Vector{Int32} = Int32[32, 64, 128, 256, 512]
    mc_resolution_labels::Vector{String} = ["32", "64", "128", "256", "512"]

    mt_subdivision::Ref{Int32} = Ref{Int32}(0)

    save_file::Vector{UInt8} = Vector{UInt8}("./model.obj" * "\0"^254)
    threshold::Ref{Float32} = Ref{Float32}(2.5f0)
end

function handle_ui!(meshing_mode::MeshingMode; ngui)
    CImGui.Begin("Meshing Mode")

    if CImGui.CollapsingHeader("Help", CIM_HEADER)
        CImGui.TextWrapped("Adjust render bbox on the main screen to select region to mesh.")
        CImGui.TextWrapped("Note that meshing operations will block until completion.")
    end

    CImGui.Separator()
    CImGui.SliderFloat("Log-density threshold", meshing_mode.threshold, 0f0, 10.0f0)

    CImGui.InputText("Save file (.obj)", pointer(meshing_mode.save_file),
        length(meshing_mode.save_file))

    CImGui.PushItemWidth(-150)
    CImGui.Combo("Meshing algorithm", meshing_mode.algorithm,
        meshing_mode.algorithm_labels, length(meshing_mode.algorithm_labels))

    is_mc = meshing_mode.algorithm[] == 0
    if is_mc
        CImGui.PushItemWidth(-150)
        CImGui.Combo("Meshing resolution", meshing_mode.mc_resolution,
            meshing_mode.mc_resolution_labels,
            length(meshing_mode.mc_resolution_labels))
    else
        CImGui.PushItemWidth(-150)
        CImGui.SliderInt("Subdivision count", meshing_mode.mt_subdivision,
            Int32(0), Int32(5))
    end

    CImGui.BeginTable("##meshing-buttons-table", 2)
    CImGui.TableNextRow()
    CImGui.TableNextColumn()

    if CImGui.Button("Mesh & Save", CImGui.ImVec2(-1, 0))
        save_file = unsafe_string(pointer(meshing_mode.save_file))
        endswith(lowercase(save_file), ".obj") ||
            (save_file = save_file * ".obj";)

        threshold = meshing_mode.threshold[]

        # TODO display error if meshing fails instead of crash
        if is_mc
            resolution_idx = 1 + meshing_mode.mc_resolution[]
            resolution::Int64 = meshing_mode.mc_resolutions[resolution_idx]

            vertices, normals, indices = Nerf.marching_cubes(
                ngui.renderer, ngui.trainer.bbox;
                threshold, mesh_resolution=resolution,
            ) do points
                Nerf.batched_density(
                    ngui.trainer.model, points; batch=256 * 256)
            end
        else
            subdivide = meshing_mode.mt_subdivision[]
            vertices, normals, indices = Nerf.marching_tetrahedra(
                ngui.renderer, ngui.trainer.bbox; threshold, subdivide,
            ) do points
                Nerf.batched_density(
                    ngui.trainer.model, points; batch=256 * 256)
            end
        end

        colors = Nerf.trace_vertex_colors(
            ngui.renderer, ngui.trainer.occupancy, ngui.trainer.bbox;
            vertices, normals,
        ) do points, directions
            ngui.trainer.model(points, directions)
        end

        Nerf.save_mesh(save_file,
            Array(vertices), Array(normals), Array(colors), Array(indices),
            nerf_offset=ngui.trainer.dataset.offset,
            nerf_scale=ngui.trainer.dataset.scale)
    end

    CImGui.TableNextColumn()
    if CImGui.Button("Go Back", CImGui.ImVec2(-1, 0))
        ngui.screen = MainScreen
    end
    CImGui.EndTable()

    CImGui.End()
end

function loop!(meshing_mode::MeshingMode; ngui)
    frame_time = update_time!(ngui.render_state)

    NeuralGraphicsGL.imgui_begin(ngui.context)
    handle_ui!(meshing_mode; ngui)

    if !is_mouse_in_ui()
        ngui.render_state.need_render |= handle_keyboard!(
            ngui.controls, ngui.renderer.camera; frame_time)
        ngui.render_state.need_render |= handle_mouse!(
            ngui.controls, ngui.renderer.camera)
    end

    NeuralGraphicsGL.set_clear_color(0.2, 0.2, 0.2, 1.0)
    NeuralGraphicsGL.clear()

    render!(ngui)
    NeuralGraphicsGL.draw(ngui.render_state.surface)

    NeuralGraphicsGL.imgui_end(ngui.context)
    glfwSwapBuffers(ngui.context.window)
    glfwPollEvents()
    return nothing
end
