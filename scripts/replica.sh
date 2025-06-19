echo "Running RIM2 on Replica"

echo "\e[1;32mrosrun neural_mapping neural_mapping_node train config/replica/replica.yaml data/Replica/room0\e[0m"
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/room0

echo "\e[1;32mrosrun neural_mapping neural_mapping_node train config/replica/replica.yaml data/Replica/office0\e[0m"
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/office0

echo "\e[1;32mrosrun neural_mapping neural_mapping_node train config/replica/replica.yaml data/Replica/office1\e[0m"
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/office1

echo "\e[1;32mrosrun neural_mapping neural_mapping_node train config/replica/replica.yaml data/Replica/office2\e[0m"
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/office2

echo "\e[1;32mrosrun neural_mapping neural_mapping_node train config/replica/replica.yaml data/Replica/office3\e[0m"
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/office3

echo "\e[1;32mrosrun neural_mapping neural_mapping_node train config/replica/replica.yaml data/Replica/office4\e[0m"
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/office4

echo "\e[1;32mrosrun neural_mapping neural_mapping_node train config/replica/replica.yaml data/Replica/room1\e[0m"
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/room1

echo "\e[1;32mrosrun neural_mapping neural_mapping_node train config/replica/replica.yaml data/Replica/room2\e[0m"
rosrun neural_mapping neural_mapping_node train src/GS-SDF/config/replica/replica.yaml src/GS-SDF/data/Replica/room2