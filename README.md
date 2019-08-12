# SpeedLimit_DL
A deep learning project to estimate the current speedlimit from a front camera video feed.

AlexNet is trained from the German Trafic Sign Dataset from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

The dataset structure should be:

./data/  
    -- train/  
        -- 00000/  
            -- 00000_00000.ppm  
            -- 00000_00001.ppm  
            -- ...  
        -- ...  
    ...  
