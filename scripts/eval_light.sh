lighteval accelerate \
     "pretrained=$1" \
     configs/lighteval_benchmark_list.txt
# or, e.g., "leaderboard|truthfulqa:mc|0|0|,leaderboard|gsm8k|3|1"