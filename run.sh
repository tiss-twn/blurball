python src/main.py --config-name=inference_blurball \
    WASB_ROOT=/home/colligo/code_local/BlurBall \
    detector.model_path=blurball_models/blurball_best \
    +input_vid=foul_1.mp4 \
    detector.step=1 \
    detector.postprocessor.score_threshold=0.7

