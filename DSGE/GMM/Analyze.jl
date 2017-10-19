d1 = readdlm("First.out")
d2 = readdlm("Second.out")
d3 = readdlm("Third.out")
d4 = readdlm("Fourth.out")
d5 = readdlm("Fifth.out")

obj = [d1[:,20] d2[:,20] d3[:,20] d4[:,20] d5[:,20]]
prettyprint(obj)
d = dstats(obj');
test = d[:,2] .> 0.001
println("proportion with more than one solution: ", mean(test))
results = zeros(100,18)
# choose the best of the runs
for i = 1:100
    ind = indmin(obj[i,:])
    if ind == 1
        results[i,:] = d1[i,1:18]
    elseif ind == 2    
        results[i,:] = d2[i,1:18]
    elseif ind == 3    
        results[i,:] = d3[i,1:18]
    elseif ind == 4   
        results[i,:] = d4[i,1:18]
    else    
        results[i,:] = d5[i,1:18]
    end    
end
dstats(results);
# choose the worst of the runs
for i = 1:100
    ind = indmax(obj[i,:])
    if ind == 1
        results[i,:] = d1[i,1:18]
    elseif ind == 2    
        results[i,:] = d2[i,1:18]
    elseif ind == 3    
        results[i,:] = d3[i,1:18]
    elseif ind == 4   
        results[i,:] = d4[i,1:18]
    else    
        results[i,:] = d5[i,1:18]
    end    
end
dstats(results);
#prettyprint(d1[15:15,:])
