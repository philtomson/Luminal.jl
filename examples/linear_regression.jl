using Luminal
using Luminal.Optimizer
using Printf

# 1. Setup Data
# Target: y = 2x + 1
X_data = Float32.(randn(1, 100))
Y_data = 2.0f0 .* X_data .+ 1.0f0

# 2. Build Model
g = Graph()
x = tensor(g, [1, 100])
y_true = tensor(g, [1, 100])

w = tensor(g, [1, 1])
b = tensor(g, [1])

# Initialize weights
g.tensors[(w.id, 1)] = Float32[0.1;;]
g.tensors[(b.id, 1)] = Float32[0.0;;]

mark_trainable!(w)
mark_trainable!(b)

# y_pred = w * x + b
y_pred = w * x + b

# Loss = Mean Squared Error
diff = y_pred - y_true
loss = (1.0f0 / 100.0f0) * sum(diff * diff, 1)

# 3. Setup Optimizer
opt = Adam(lr=0.01f0)
grads_mapping = backward(loss)

# 4. Training Loop
println("Starting Training Linear Regression...")
println("Initial w: ", g.tensors[(w.id, 1)][1], ", b: ", g.tensors[(b.id, 1)][1])

for epoch in 1:200
    inputs = Dict(x.id => X_data, y_true.id => Y_data)
    
    # Forward & Backward
    # We need loss and all gradients
    grad_ids = [v.id for v in values(grads_mapping)]
    res = execute(g, [loss.id, grad_ids...], inputs, CPUDevice())
    
    # Update
    actual_grads = Dict(pid => res[gt.id] for (pid, gt) in grads_mapping)
    step!(opt, g, actual_grads)
    
    if epoch % 20 == 0
        @printf("Epoch %3d | Loss: %8.6f | w: %6.4f | b: %6.4f\n", 
                epoch, res[loss.id][1], g.tensors[(w.id, 1)][1], g.tensors[(b.id, 1)][1])
    end
end

println("Final Results:")
@printf("w: %6.4f (Target: 2.0), b: %6.4f (Target: 1.0)\n", 
        g.tensors[(w.id, 1)][1], g.tensors[(b.id, 1)][1])
