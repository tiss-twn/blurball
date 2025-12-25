#!/bin/bash


export WASB_ROOT=$(realpath src/configs/global)

#python src/main.py --config-name=inference_blurball \
#    WASB_ROOT=/home/colligo/code_local/BlurBall \
#    detector.model_path=blurball_models/blurball_best \
#    +input_vid=foul_1.mp4 \
#    detector.step=1 \
#    detector.postprocessor.score_threshold=0.5

# write a loop that does 00.mp4 - 03.mp4
for i in {00..03}; do
    python src/main.py --config-name=inference_blurball \
        WASB_ROOT=/home/colligo/code_local/BlurBall \
        detector.model_path=blurball_models/blurball_best \
        +input_vid=${i}.mp4 \
        output_dir=output_${i} \
        detector.step=1 \
        detector.postprocessor.score_threshold=0.5
done


