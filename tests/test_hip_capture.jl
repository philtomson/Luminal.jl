# Executors that copy must be capturable in a HIP graph. `copyto!` between two device arrays is a memcpy on the null stream, which aborts a
# capture (hipErrorStreamCaptureUnsupported); the compiler aliases most Reshape / Contiguous / Slice nodes away, but the e-graph search's candidates
# keep some, and a capture-mode measured search of a decode graph crashed on them. Here the executors run directly inside a capture.
using Test
using Luminal
using AMDGPU

if !(Luminal.get_device() isa Luminal.AMDDevice)
    @info "no AMD GPU: skipping the HIP capture test"
else
    @testset "copying executors capture and replay in a HIP graph" begin
        dev = Luminal.get_device()
        a = Luminal.to_device(randn(Float32, 8, 6, 2), dev)
        host = Array(a)
        outs = Dict(:reshape => AMDGPU.zeros(Float32, 4, 12, 2), :contiguous => AMDGPU.zeros(Float32, 8, 6, 2), :slice => AMDGPU.zeros(Float32, 3, 6, 2),
                    :slice_mid => AMDGPU.zeros(Float32, 8, 2, 2))
        function work()
            Luminal.execute_op!(outs[:reshape], Luminal.Reshape([4, 12, 2]), a)
            Luminal.execute_op!(outs[:contiguous], Luminal.Contiguous(), a)
            Luminal.execute_slice!(outs[:slice], a, [(2, 5), (0, 6), (0, 2)])               # rows 3:5
            Luminal.execute_slice!(outs[:slice_mid], a, [(0, 8), (1, 3), (0, 2)])           # columns 2:3: a strided view
            return nothing
        end
        work(); AMDGPU.synchronize()                                                          # warm-up (compiles the kernels)
        for o in values(outs); fill!(o, 0f0); end
        g = Luminal.Device._capture_hip_graph(work)                                           # throws if any executor memcpys
        for o in values(outs); fill!(o, 0f0); end
        for _ in 1:2
            Luminal.Device._launch_hip_graph(g); AMDGPU.synchronize()
            @test Array(outs[:reshape]) == Base.reshape(host, 4, 12, 2)
            @test Array(outs[:contiguous]) == host
            @test Array(outs[:slice]) == host[3:5, :, :]
            @test Array(outs[:slice_mid]) == host[:, 2:3, :]
        end
    end
end
