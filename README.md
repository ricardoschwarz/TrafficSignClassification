# TrafficSignClassification
A deep learning project that aims to classify visible traffic signs. Aims to be extended for live detection and classification.

CNNs are trained from the German Trafic Sign Dataset from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

The dataset structure should be:

<pre>
./data/  
-- train/  
        -- 00000/  
            -- 00000_00000.ppm  
            -- 00000_00001.ppm  
            -- ...  
        -- ...  
</pre>

Labels of all classes:

| **Nr**    | **Label**                               |
| ----- | -------------------------------------------- |
| 00000 | SpeedLimit = 20                              |
| 00001 | SpeedLimit = 30                              |
| 00002 | SpeedLimit = 50                              |
| 00003 | SpeedLimit = 60                              |
| 00004 | SpeedLimit = 70                              |
| 00005 | SpeedLimit = 80                              |
| 00006 | End of SpeedLimit = 80                       |
| 00007 | End of SpeedLimit = 100                      |
| 00008 | SpeedLimit = 120                             |
| 00009 | No overtaking allowed for cars               |
| 00010 | No overtaking allowed for trucks             |
| 00011 | Right of way at next junction                |
| 00012 | Right of way on this road                    |
| 00013 | Yield right of way                           |
| 00014 | Stop. Yield right of way                     |
| 00015 | All vehicles banned for this road            |
| 00016 | All trucks banned for this road              |
| 00017 | No Entry!                                    |
| 00018 | Danger Spot                                  |
| 00019 | Sharp Left Corner                            |
| 00020 | Sharp Right Corner                           |
| 00021 | Double Curve                                 |
| 00022 | Uneven Road                                  |
| 00023 | Slip Hazard                                  |
| 00024 | Road narrows                                 |
| 00025 | Roadworks                                    |
| 00026 | Traffic Light                                |
| 00027 | Pedestrians                                  |
| 00028 | Children                                     |
| 00029 | Cyclists                                     |
| 00030 | Slipperiness                                 |
| 00031 | Deer Path                                    |
| 00032 | End of all constraints                       |
| 00033 | Prescribed driving direction right           |
| 00034 | Prescribed driving direction left            |
| 00035 | Prescribed driving direction ahead           |
| 00036 | Prescribed driving direction ahead and right |
| 00037 | Prescribed driving direction ahead and left  |
| 00038 | Prescribed passing at the right side         |
| 00039 | Prescribed passing at the left side          |
| 00040 | Roundabout                                   |
| 00041 | Overtaking is now allowed                    |
| 00042 | Overtaking is now allowed for trucks         |

