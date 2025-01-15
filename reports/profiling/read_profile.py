import pstats
import os
p = pstats.Stats('reports/profiling/profile.txt')
#p.sort_stats('cumulative').print_stats(20)
p.sort_stats('tottime').print_stats(20)

# To visualize the profile, we can use the SnakeViz package. snakevis reports/profiling/profile.prof