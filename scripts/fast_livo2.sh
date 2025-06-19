echo "Running RIM2 on FAST_LIVO2"

rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/fast_livo/drive.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/drive/fast_livo2_drive.bag
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/fast_livo/red_bird.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/red_bird2/fast_livo2_red_bird2.bag
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/fast_livo/campus.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/campus/fast_livo2_campus.bag
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/fast_livo/station.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/2_dao_fast/fast_livo2_2_dao_fast.bag
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/fast_livo/cbd.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/cbd/fast_livo2_000.bag
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/fast_livo/sysu.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/sysu_large_scale_2023-11-28-15-19-20/fast_livo2_large_scale_2023-11-28-15-19-20_filter.bag
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/fast_livo/culture01.yaml src/GS-SDF/data/FAST_LIVO2_RIM_Datasets/culture01/fast_livo2_culture01.bag
