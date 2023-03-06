# This Code based on "Power Consumption Characterization, Modeling and Estimation of Electric Vehicles"
# energy = integral(IVdt) = integral(Fds)
# I , V -> EV motor Voltage and Current.
# F     -> traction force, s-> moving distance
# P = F * (ds/dt) = Fv = Fr+Fa+Fg+Fi+Fb ≈ Fr+Fg+Fi ≈ (α+βsinθ+γa)mv

# Fr -> rolling     resistance
# Fa -> aerodynamic resistance
# Fg -> gradient    resistance
# Fi -> inertia     resistance
# Fb -> brake force proveded by hydraulic brake

# Fr ∝ C_{rr}*W, Fa ∝ 1/2 *ρCdAv2 , Fg ∝ W sinθ, Fi ∝ ma
# p -> rPth, cd -> rPth, A -> rPtn, v^2 ->
# in reference thesis, Fa is ignored because of vehicle's limited speed(34km/h)
# but our vehicle is for high speed racing, so Fa must considered.

# α is rolling     resistance
# β is gradient    resistance
# γ is inertia     resistance
# δ is aerodynamic resistance


# (α+βsinθ+γa+)mv + δv^2 
# (α+βsinθ+γa)v +δv^2

# θ, a(accel), v(velocity).
# btw, beta -> ignore, a -> ignore..

# m -> in alpha beta, gamma, 