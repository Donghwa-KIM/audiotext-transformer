EPOCHS=10
BATCH_SIZE=64
DIM_MODEL=64
ATTENTION_DROPOUT=0.2
EMBED_DROPOUT=0.2
ATTENTION_DROPOUT_AUDIO=0.0
OUT_DROPOUT=0.1
RESAMPLE_RATE=8000
N_FFT_SIZE=400
MERGE_HOW=average


python train.py --lr=0.00001 \
    --d_model=${DIM_MODEL} \
    --attn_dropout=${ATTENTION_DROPOUT} \
    --embed_dropout=${EMBED_DROPOUT} \
    --attn_dropout_a=${ATTENTION_DROPOUT_AUDIO} \
    --out_dropout=${OUT_DROPOUT} \
    --resample_rate=${RESAMPLE_RATE} \
    --n_fft_size=${N_FFT_SIZE} \
    --merge_how=${MERGE_HOW} \
    --num_epochs=${EPOCHS} \
    --batch_size=${BATCH_SIZE}
