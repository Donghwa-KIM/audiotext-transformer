

EPOCHS=10
BATCH_SIZE=64

for DIM_MODEL in 64 128 256
do
    for ATTENTION_DROPOUT in 0.2 0.0 0.1 
    do
        for EMBED_DROPOUT in 0.2 0.0 0.1 
        do
            for ATTENTION_DROPOUT_AUDIO in 0.0 0.1 0.2
            do
                for OUT_DROPOUT in 0.1 0.0 0.2
                do
                    for RESAMPLE_RATE in 16000 48000 8000 20
                    do
                        for N_FFT_SIZE in 400 1024 200
                        do
                            for MERGE_HOW in average last_hidden
                            do
                            echo "
                                --DIM_MODEL=$DIM_MODEL
                                --ATTENTION_DROPOUT=$ATTENTION_DROPOUT
                                --EMBED_DROPOUT=$EMBED_DROPOUT
                                --ATTENTION_DROPOUT_AUDIO=$ATTENTION_DROPOUT_AUDIO
                                --OUT_DROPOUT=${OUT_DROPOUT}
                                --RESAMPLE_RATE==$RESAMPLE_RATE
                                --N_FFT_SIZE=$N_FFT_SIZE
                                --MERGE_HOW=$MERGE_HOW
                                --NUM_EPOCHS=${EPOCHS}
                                --BATCH_SIZE=${BATCH_SIZE}
                                "
                            python main.py --lr=0.00001 \
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

                            python main.py --do_audio \
                                --lr=0.001 \
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



                            done
                        done
                    done
                done
            done
        done
    done
done
