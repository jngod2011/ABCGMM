using Plots
pyplot()
data = readdlm("EstimationResults.out")
tau = data[1,:]

plot(reuse=false)
constant = data[6,:]
lb = data[8,:]
ub = data[9,:]
plot!(tau, [constant constant], fillrange=[lb ub], fillalpha=0.3, c=:green, legend=:none)
xticks!((1:9)/10)
savefig("mean2yearscollege.svg")

plot(reuse=false)
educ = data[2,:]
lb = data[4,:]
ub = data[5,:]
plot!(tau, [educ educ], fillrange=[lb ub], fillalpha=0.3, c=:green, legend=:none)
#ylims!(0.08, 0.22)
xticks!((1:9)/10)
savefig("educ.svg")

plot(reuse=false)
black = data[10,:]
lb = data[12,:]
ub = data[13,:]
plot!(tau, [black black], fillrange=[lb ub], fillalpha=0.3, c=:green, legend=:none)
#ylims!(0.08, 0.22)
xticks!((1:9)/10)
savefig("black.svg")

