Pooling layers.md


Max pooling just takes the Max value inside a grid of width and length stride
slide the max pooling later over by the stride i.e. no overlap between grides

1 2 3 4
5 6 7 8 
9 8 7 6
5 4 3 2

Max pooling this with stride 2
Becomes, Reduces 16 -> 4
6 8
9 7


Non-Global pooling layers reduce the size:
Becomes, Reduces 16 -> 4
3.5 5.5
6.5 4.5

Global Average pooling:
    Dont specify window size nor stride 
    More extreme 
    add global nodes 
    80
    Then divide by number of nodes
    Do this on a per feature map level
    Each Feature map can be reduced to a SINGLE value
    Takes multp