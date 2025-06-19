if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

model=$1

# find parent_path according to $0
parent_path=$(dirname "$0")
python $parent_path/metrics.py -m $model/train/color
python $parent_path/metrics.py -m $model/eval/color
