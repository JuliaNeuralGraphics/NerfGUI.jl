mutable struct OccupancyView
    voxels::GL.Voxels
    last_update_step::Int
    n_levels::Int
end

OccupancyView() = OccupancyView(GL.Voxels(Float32[]), 0, 0)

function reset!(oc::OccupancyView)
    oc.voxels.n_voxels == 0 && oc.n_levels == 0 && return nothing
    GL.update!(oc.voxels, Float32[])
    oc.n_levels = 0
    return nothing
end

function update!(
    oc::OccupancyView, occupancy::Nerf.OccupancyGrid;
    n_levels::Integer, update_step::Integer,
)
    oc.last_update_step == update_step &&
        oc.n_levels == n_levels && return nothing

    reset!(oc)
    min_density, max_density = 0f0, occupancy.mean_density * 2f0
    density, binary = Array(occupancy.density), Array(occupancy.binary)

    resolution = UInt32(Nerf.get_resolution(occupancy))
    level_length = Nerf.offset(occupancy, 1)
    index_range = UInt32(0):UInt32(8):UInt32(level_length - 1)

    data = Float32[]
    for level in UInt32(0):UInt32(n_levels)
        offset = Nerf.offset(occupancy, level)
        diameter = Nerf.get_voxel_diameter(occupancy, level)

        for i in index_range
            point = Nerf.index_to_point(i, resolution, level)
            if Nerf.is_occupied(binary, point, resolution, level)
                density01 = clamp(
                    density[i + offset + 1], min_density, max_density)
                density01 = (density01 - min_density) / (max_density - min_density)
                append!(data, point..., density01, diameter)
            end
        end
    end

    GL.update!(oc.voxels, data)
    oc.last_update_step = update_step
    oc.n_levels = n_levels
    return nothing
end

function GL.draw(oc::OccupancyView, projection, look_at)
    GL.enable_blend()
    GL.draw_instanced(oc.voxels, projection, look_at)
    GL.disable_blend()
    return nothing
end
