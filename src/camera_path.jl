Base.@kwdef mutable struct CameraPath
    keyframes::Vector{Nerf.CameraKeyframe} = []
    current_time::Float32 = 0f0
    current_step::Int64 = 0

    gui_lines::Vector{GL.Line} = []
    line_program::GL.ShaderProgram = GL.get_program(GL.Line)
end

function Base.push!(p::CameraPath, c::Nerf.Camera)
    new_keyframe = Nerf.CameraKeyframe(c)
    push!(p.keyframes, new_keyframe)
    length(p.keyframes) == 1 && return nothing

    from = p.keyframes[end - 1].t
    push!(p.gui_lines, GL.Line(from, new_keyframe.t; program=p.line_program))
    nothing
end

function Base.getindex(p::CameraPath, i::Integer)
    isempty(p.keyframes) &&
        error("Tried accessing an empty CameraPath at `$i` index.")
    p.keyframes[clamp(i, 1, length(p.keyframes))]
end

Base.length(p::CameraPath) = length(p.keyframes)

Base.isempty(p::CameraPath) = isempty(p.keyframes)

function get_time_step(p::CameraPath, n_steps::Integer)
    1f0 / (n_steps * (length(p) - 1))
end

function advance!(p::CameraPath, δ::Float32)
    p.current_time += δ
    p.current_step += 1
end

is_done(p::CameraPath) = p.current_time ≥ 1f0

function reset_time!(p::CameraPath)
    p.current_time = 0f0
    p.current_step = 0
end

function Base.empty!(p::CameraPath)
    p.current_time = 0f0
    p.current_step = 0

    GL.delete!.(p.gui_lines; with_program=false)
    empty!(p.keyframes)
    empty!(p.gui_lines)
end

"""
# Arguments:

- `t::Float32`: Time value in `[0, 1]` range. Goes through all keyframes.
"""
function eval(p::CameraPath)
    t = p.current_time
    t = t * (length(p) - 1)
    idx = floor(Int, t) + 1
    Nerf.spline(t - floor(t), p[idx - 1], p[idx], p[idx + 1], p[idx + 2])
end

function GL.draw(p::CameraPath, P, L)
    isempty(p.gui_lines) && return nothing
    for l in p.gui_lines
        GL.draw(l, P, L)
    end
    nothing
end
